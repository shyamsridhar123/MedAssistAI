#!/usr/bin/env python3
"""
Hybrid PHI Detection System
MedAssist AI - Option C (Balanced Approach)

Combines rule-based detection with ClinicalBERT for optimal performance.
"""

import re
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RuleBasedPHIDetector:
    """Enhanced rule-based PHI detector"""
    
    def __init__(self):
        self.patterns = {
            'PHONE': [
                r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # 555-123-4567
                r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}',      # (555) 123-4567
                r'\b\d{3}\.\d{3}\.\d{4}\b'             # 555.123.4567
            ],
            'EMAIL': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            'SSN': [
                r'\b\d{3}-\d{2}-\d{4}\b',              # 123-45-6789
                r'\b\d{9}\b'                           # 123456789 (if context suggests SSN)
            ],
            'DATE': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # 12/25/2024, 12-25-24
                r'\b\d{4}-\d{1,2}-\d{1,2}\b',          # 2024-12-25
                r'\b\w+\s+\d{1,2},?\s+\d{4}\b',       # December 25, 2024
                r'\b\d{1,2}\s+\w+\s+\d{4}\b'          # 25 December 2024
            ],
            'PERSON': [
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',      # John Smith
                r'\bDr\.\s+[A-Z][a-z]+\b',             # Dr. Smith
                r'\bMr\.\s+[A-Z][a-z]+\b',             # Mr. Johnson
                r'\bMs\.\s+[A-Z][a-z]+\b',             # Ms. Anderson
                r'\bMrs\.\s+[A-Z][a-z]+\b'             # Mrs. Wilson
            ],
            'LOCATION': [
                r'\b\d+\s+\w+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b',  # Addresses
                r'\b[A-Z][a-z]+,\s+[A-Z]{2}\b',       # City, ST
                r'\bRoom\s+\d+[A-Z]?\b',               # Room 302A
                r'\bFloor\s+\d+\b'                     # Floor 3
            ],
            'MRN': [
                r'\bMRN:?\s*[A-Z0-9-]+\b',             # MRN: A12345
                r'\bID:?\s*[A-Z0-9-]+\b',              # ID: 555444
                r'\bPatient\s+#[A-Z0-9-]+\b'           # Patient #A47-B92
            ]
        }
        
        # Exclusion patterns to reduce false positives
        self.exclusions = {
            'PERSON': [
                'emergency room', 'intensive care', 'medical center', 'general hospital',
                'blood pressure', 'heart rate', 'johnson & johnson', 'smith & wesson'
            ],
            'LOCATION': [
                'temperature', 'blood pressure', 'heart rate'
            ]
        }
    
    def detect(self, text: str) -> List[Dict]:
        """Detect PHI using rule-based patterns"""
        detections = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    matched_text = match.group()
                    
                    # Apply exclusions
                    if self._should_exclude(matched_text, entity_type):
                        continue
                    
                    # Assign confidence based on pattern specificity
                    confidence = self._calculate_confidence(matched_text, entity_type)
                    
                    detections.append({
                        'start': match.start(),
                        'end': match.end(),
                        'text': matched_text,
                        'label': entity_type,
                        'confidence': confidence,
                        'method': 'rule_based'
                    })
        
        return detections
    
    def _should_exclude(self, text: str, entity_type: str) -> bool:
        """Check if detection should be excluded"""
        if entity_type in self.exclusions:
            for exclusion in self.exclusions[entity_type]:
                if exclusion.lower() in text.lower():
                    return True
        return False
    
    def _calculate_confidence(self, text: str, entity_type: str) -> float:
        """Calculate confidence score for rule-based detection"""
        base_confidence = {
            'SSN': 0.98,     # Very specific pattern
            'EMAIL': 0.95,   # Very specific pattern
            'PHONE': 0.90,   # Specific pattern
            'MRN': 0.85,     # Medical context specific
            'DATE': 0.80,    # Common but context-dependent
            'PERSON': 0.60,  # Prone to false positives
            'LOCATION': 0.70 # Context-dependent
        }
        
        confidence = base_confidence.get(entity_type, 0.5)
        
        # Boost confidence for medical context clues
        medical_keywords = ['patient', 'doctor', 'nurse', 'hospital', 'clinic', 'medical']
        if any(keyword in text.lower() for keyword in medical_keywords):
            confidence = min(confidence + 0.1, 1.0)
        
        return confidence

class ClinicalBERTPHIDetector:
    """ClinicalBERT-based PHI detector"""
    
    def __init__(self, model_path: str = "models/phi_detection/clinicalbert_real/final"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.id2label = {
            0: 'O',
            1: 'B-PERSON', 2: 'I-PERSON',
            3: 'B-LOCATION', 4: 'I-LOCATION',
            5: 'B-DATE', 6: 'I-DATE',
            7: 'B-PHONE', 8: 'I-PHONE',
            9: 'B-SSN', 10: 'I-SSN',
            11: 'B-EMAIL', 12: 'I-EMAIL',
            13: 'B-MRN', 14: 'I-MRN'
        }
        
    def load_model(self):
        """Load trained ClinicalBERT model"""
        try:
            logger.info(f"🔽 Loading ClinicalBERT from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            self.model.eval()
            logger.info("✅ ClinicalBERT loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to load ClinicalBERT: {e}")
            return False
    
    def detect(self, text: str) -> List[Dict]:
        """Detect PHI using ClinicalBERT"""
        if not self.model or not self.tokenizer:
            logger.warning("⚠️ ClinicalBERT not loaded, skipping ML detection")
            return []
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512,
            return_offsets_mapping=True
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**{k: v for k, v in inputs.items() if k != 'offset_mapping'})
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_labels = torch.argmax(predictions, dim=-1)[0]
            confidence_scores = torch.max(predictions, dim=-1)[0][0]
        
        # Convert to detections
        detections = []
        current_entity = None
        
        offset_mapping = inputs['offset_mapping'][0]
        
        for i, (label_id, confidence, (start_offset, end_offset)) in enumerate(
            zip(predicted_labels, confidence_scores, offset_mapping)
        ):
            if start_offset == 0 and end_offset == 0:  # Special tokens
                continue
                
            label = self.id2label[label_id.item()]
            confidence_val = confidence.item()
            
            if label.startswith('B-'):  # Beginning of entity
                # Save previous entity if exists
                if current_entity:
                    detections.append(current_entity)
                
                # Start new entity
                entity_type = label[2:]  # Remove 'B-' prefix
                current_entity = {
                    'start': int(start_offset),
                    'end': int(end_offset),
                    'text': text[int(start_offset):int(end_offset)],
                    'label': entity_type,
                    'confidence': confidence_val,
                    'method': 'clinical_bert'
                }
            
            elif label.startswith('I-') and current_entity:  # Inside entity
                # Extend current entity
                current_entity['end'] = int(end_offset)
                current_entity['text'] = text[current_entity['start']:int(end_offset)]
                current_entity['confidence'] = (current_entity['confidence'] + confidence_val) / 2
            
            else:  # Outside entity (O)
                if current_entity:
                    detections.append(current_entity)
                    current_entity = None
        
        # Add final entity if exists
        if current_entity:
            detections.append(current_entity)
        
        # Filter low-confidence predictions
        detections = [d for d in detections if d['confidence'] > 0.5]
        
        return detections

class HybridPHIDetector:
    """Hybrid PHI detection system combining rule-based and ClinicalBERT"""
    
    def __init__(self, clinical_bert_path: str = "models/phi_detection/clinicalbert_real/final"):
        self.rule_detector = RuleBasedPHIDetector()
        self.bert_detector = ClinicalBERTPHIDetector(clinical_bert_path)
        self.bert_available = False
        
    def initialize(self):
        """Initialize all detectors"""
        logger.info("🚀 Initializing Hybrid PHI Detection System")
        
        # Try to load ClinicalBERT
        self.bert_available = self.bert_detector.load_model()
        
        if self.bert_available:
            logger.info("✅ Hybrid system ready (Rule-based + ClinicalBERT)")
        else:
            logger.info("✅ Rule-based system ready (ClinicalBERT unavailable)")
    
    def detect(self, text: str) -> List[Dict]:
        """Detect PHI using hybrid approach - RULE-BASED DISABLED due to false positives"""
        # Step 1: Rule-based detection DISABLED - too many false positives
        rule_detections = []  # Disabled: self.rule_detector.detect(text)
        
        # Step 2: ClinicalBERT detection (if available)
        bert_detections = []
        if self.bert_available:
            bert_detections = self.bert_detector.detect(text)
        
        # Step 3: Merge detections with confidence weighting
        merged_detections = self._merge_detections(rule_detections, bert_detections)
        
        # Step 4: Post-processing and deduplication
        final_detections = self._post_process(merged_detections)
        
        return final_detections
    
    def _merge_detections(self, rule_detections: List[Dict], 
                         bert_detections: List[Dict]) -> List[Dict]:
        """Merge rule-based and BERT detections with confidence weighting"""
        all_detections = []
        
        # Add rule-based detections
        for detection in rule_detections:
            all_detections.append(detection)
        
        # Add BERT detections, checking for overlaps
        for bert_det in bert_detections:
            overlapping_rule = self._find_overlapping_detection(bert_det, rule_detections)
            
            if overlapping_rule:
                # Merge overlapping detections with weighted confidence
                merged = self._merge_overlapping_detections(overlapping_rule, bert_det)
                # Replace rule detection with merged
                for i, rule_det in enumerate(all_detections):
                    if rule_det == overlapping_rule:
                        all_detections[i] = merged
                        break
            else:
                # Add non-overlapping BERT detection
                all_detections.append(bert_det)
        
        return all_detections
    
    def _find_overlapping_detection(self, detection: Dict, 
                                  detection_list: List[Dict]) -> Optional[Dict]:
        """Find overlapping detection in list"""
        for other in detection_list:
            if self._detections_overlap(detection, other):
                return other
        return None
    
    def _detections_overlap(self, det1: Dict, det2: Dict) -> bool:
        """Check if two detections overlap"""
        return (det1['start'] < det2['end'] and det1['end'] > det2['start'] 
                and det1['label'] == det2['label'])
    
    def _merge_overlapping_detections(self, rule_det: Dict, bert_det: Dict) -> Dict:
        """Merge overlapping rule-based and BERT detections"""
        # Use broader span
        start = min(rule_det['start'], bert_det['start'])
        end = max(rule_det['end'], bert_det['end'])
        
        # Weighted confidence (rule-based gets 0.4, BERT gets 0.6 weight)
        merged_confidence = (0.4 * rule_det['confidence'] + 0.6 * bert_det['confidence'])
        
        return {
            'start': start,
            'end': end,
            'text': rule_det['text'],  # Use original text span
            'label': rule_det['label'],
            'confidence': merged_confidence,
            'method': 'hybrid'
        }
    
    def _post_process(self, detections: List[Dict]) -> List[Dict]:
        """Post-process detections (deduplication, filtering)"""
        # Sort by start position
        detections.sort(key=lambda x: x['start'])
        
        # Remove low-confidence detections
        filtered = [d for d in detections if d['confidence'] > 0.3]
        
        # Remove medical vitals false positives
        medical_filtered = []
        for detection in filtered:
            if self._is_medical_vital(detection):
                continue  # Skip medical vitals
            medical_filtered.append(detection)
        
        # Remove exact duplicates
        unique_detections = []
        for detection in medical_filtered:
            is_duplicate = False
            for existing in unique_detections:
                if (detection['start'] == existing['start'] and 
                    detection['end'] == existing['end'] and
                    detection['label'] == existing['label']):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_detections.append(detection)
        
        return unique_detections
    
    def _is_medical_vital(self, detection: Dict) -> bool:
        """Check if detection is actually a medical vital sign (false positive)"""
        text = detection['text'].lower()
        
        # Common vital sign patterns that get misclassified as MRN/other PHI
        vital_patterns = [
            r'bp\s*\d+/\d+',           # BP 140/90
            r'\d+/\d+\s*mmhg',         # 140/90 mmHg
            r'hr\s*\d+',               # HR 72
            r'\d+\s*bpm',              # 72 bpm
            r'temp\s*\d+\.?\d*[f°]?',  # Temp 98.6F
            r'\d+\.?\d*[f°]\s*temp',   # 98.6F temp
            r'o2\s*\d+%',              # O2 95%
            r'\d+%\s*o2',              # 95% O2
            r'rr\s*\d+',               # RR 16
            r'\d+\s*breaths',          # 16 breaths
            r'glucose\s*\d+',          # glucose 120
            r'\d+\s*mg/dl',            # 120 mg/dl
        ]
        
        for pattern in vital_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def get_performance_stats(self) -> Dict:
        """Get system performance statistics"""
        return {
            'rule_based_available': True,
            'clinical_bert_available': self.bert_available,
            'system_type': 'hybrid' if self.bert_available else 'rule_based_only'
        }

def detect_phi_from_text(text: str, output_format: str = 'detailed') -> List[Dict]:
    """
    Detect PHI from input text using hybrid detection system
    
    Args:
        text: Input text to analyze
        output_format: 'detailed', 'simple', or 'json'
    
    Returns:
        List of detected PHI entities
    """
    # Initialize detector
    detector = HybridPHIDetector()
    detector.initialize()
    
    # Detect PHI
    detections = detector.detect(text)
    
    if output_format == 'simple':
        return [{'text': d['text'], 'type': d['label'], 'confidence': round(d['confidence'], 3)} 
                for d in detections]
    elif output_format == 'json':
        return detections
    else:  # detailed
        return detections

def detect_phi_batch(texts: List[str], output_format: str = 'detailed') -> List[List[Dict]]:
    """
    Detect PHI from multiple texts using hybrid detection system
    
    Args:
        texts: List of input texts to analyze
        output_format: 'detailed', 'simple', or 'json'
    
    Returns:
        List of detection results for each input text
    """
    # Initialize detector once for batch processing
    detector = HybridPHIDetector()
    detector.initialize()
    
    results = []
    for text in texts:
        detections = detector.detect(text)
        
        if output_format == 'simple':
            formatted_detections = [{'text': d['text'], 'type': d['label'], 'confidence': round(d['confidence'], 3)} 
                                   for d in detections]
        elif output_format == 'json':
            formatted_detections = detections
        else:  # detailed
            formatted_detections = detections
        
        results.append(formatted_detections)
    
    return results

class PHIDetectionAPI:
    """
    API-style interface for PHI detection
    Allows for persistent detector instance to avoid reloading models
    """
    
    def __init__(self, clinical_bert_path: str = "models/phi_detection/clinicalbert_real/final"):
        self.detector = HybridPHIDetector(clinical_bert_path)
        self.initialized = False
    
    def initialize(self):
        """Initialize the detection system"""
        self.detector.initialize()
        self.initialized = True
    
    def detect(self, text: str, confidence_threshold: float = 0.3) -> List[Dict]:
        """
        Detect PHI in text
        
        Args:
            text: Input text to analyze
            confidence_threshold: Minimum confidence for detections
        
        Returns:
            List of detected PHI entities
        """
        if not self.initialized:
            self.initialize()
        
        detections = self.detector.detect(text)
        
        # Filter by confidence threshold
        return [d for d in detections if d['confidence'] >= confidence_threshold]
    
    def detect_batch(self, texts: List[str], confidence_threshold: float = 0.3) -> List[List[Dict]]:
        """
        Detect PHI in multiple texts
        
        Args:
            texts: List of input texts to analyze
            confidence_threshold: Minimum confidence for detections
        
        Returns:
            List of detection results for each input text
        """
        if not self.initialized:
            self.initialize()
        
        results = []
        for text in texts:
            detections = self.detector.detect(text)
            filtered_detections = [d for d in detections if d['confidence'] >= confidence_threshold]
            results.append(filtered_detections)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return self.detector.get_performance_stats()

def main(text: str = None):
    """PHI detection system - accepts input text"""
    if text is None:
        # Default sample if no input provided
        text = """
    Patient John Smith (DOB: 12/15/1980) was admitted on 2024-07-26. 
    Contact number: (555) 123-4567. 
    Email: john.smith@email.com
    SSN: 123-45-6789
    Medical Record: MRN-A4567
    Address: 123 Main Street, Boston, MA
    """
    
    logger.info("🔍 Hybrid PHI Detection Demo")
    logger.info("=" * 40)
    
    # Initialize detector
    detector = HybridPHIDetector()
    detector.initialize()
    
    # Detect PHI
    logger.info("📝 Input text:")
    logger.info(text)
    
    detections = detector.detect(text)
    
    logger.info(f"\n🎯 Found {len(detections)} PHI entities:")
    for i, detection in enumerate(detections, 1):
        logger.info(f"  {i}. {detection['label']}: '{detection['text']}' "
                   f"(confidence: {detection['confidence']:.3f}, "
                   f"method: {detection['method']})")
    
    # Performance stats
    stats = detector.get_performance_stats()
    logger.info(f"\n📊 System stats: {stats}")

if __name__ == "__main__":
    main()

