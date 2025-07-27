#!/usr/bin/env python3
"""
emergency_phi_bootstrap.py - Emergency PHI detection bootstrap for MedAssist AI

This script creates synthetic PHI-annotated training data and builds an initial
PHI detection model to bridge the gap until real clinical datasets are approved.
"""

import re
import random
import json
import pandas as pd
from typing import List, Tuple, Dict
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmergencyPHIBootstrap:
    """Emergency PHI detection system using synthetic data"""
    
    def __init__(self):
        # HIPAA 18 identifiers patterns
        self.phi_patterns = {
            'PERSON': [
                'John Smith', 'Mary Johnson', 'David Wilson', 'Sarah Davis', 'Michael Brown',
                'Jennifer Garcia', 'Robert Martinez', 'Lisa Anderson', 'William Taylor', 'Karen White'
            ],
            'LOCATION': [
                '123 Main Street', '456 Oak Avenue', '789 Pine Road', '321 Elm Drive',
                'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio'
            ],
            'DATE': [
                '2023-05-15', '04/12/2024', 'January 3, 2023', 'March 22, 2024',
                '12/08/2023', '07-14-2024', 'February 28, 2023'
            ],
            'PHONE': [
                '(555) 123-4567', '555-987-6543', '(444) 555-0123', '333.444.5555',
                '555 123 4567', '(666) 789-0123'
            ],
            'EMAIL': [
                'john.smith@email.com', 'patient@clinic.org', 'doctor@hospital.net',
                'mary.j@provider.com', 'contact@medical.edu'
            ],
            'SSN': [
                '123-45-6789', '987-65-4321', '555-44-3333', '111-22-3333'
            ],
            'MRN': [
                'MRN-12345678', 'MR#987654', 'ID:555444', 'Record-123456'
            ]
        }
        
        # Medical text templates for synthetic notes
        self.medical_templates = [
            "Patient {PERSON} visited on {DATE} for chest pain. Contact at {PHONE}.",
            "Follow-up appointment for {PERSON} scheduled. Lives at {LOCATION}. DOB: {DATE}.",
            "{PERSON} admitted to ICU on {DATE}. Emergency contact: {PHONE}. MRN: {MRN}.",
            "Discharge summary for patient {PERSON}. Address: {LOCATION}. SSN: {SSN}.",
            "Lab results for {PERSON} available. Email sent to {EMAIL} on {DATE}."
        ]
    
    def generate_synthetic_phi_note(self) -> Tuple[str, List[Dict]]:
        """Generate a synthetic clinical note with PHI annotations"""
        template = random.choice(self.medical_templates)
        annotations = []
        
        # Replace PHI placeholders with actual values and track positions
        note = template
        for phi_type, examples in self.phi_patterns.items():
            placeholder = f"{{{phi_type}}}"
            if placeholder in note:
                phi_value = random.choice(examples)
                start_pos = note.find(placeholder)
                note = note.replace(placeholder, phi_value, 1)
                end_pos = start_pos + len(phi_value)
                
                annotations.append({
                    'start': start_pos,
                    'end': end_pos,
                    'text': phi_value,
                    'label': phi_type
                })
        
        return note, annotations
    
    def create_training_dataset(self, num_samples: int = 1000) -> pd.DataFrame:
        """Create synthetic PHI-annotated training dataset"""
        logger.info(f"🔥 Generating {num_samples} synthetic PHI-annotated clinical notes...")
        
        training_data = []
        for i in range(num_samples):
            note, annotations = self.generate_synthetic_phi_note()
            training_data.append({
                'id': f'synthetic_{i:06d}',
                'text': note,
                'annotations': json.dumps(annotations),
                'phi_count': len(annotations)
            })
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} samples...")
        
        df = pd.DataFrame(training_data)
        logger.info(f"✅ Created synthetic training dataset: {len(df)} samples")
        return df
    
    def rule_based_phi_detector(self, text: str) -> List[Dict]:
        """Simple rule-based PHI detection"""
        detections = []
        
        # Phone number patterns
        phone_pattern = r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|\(\d{3}\)\s?\d{3}[-.\s]?\d{4}'
        for match in re.finditer(phone_pattern, text):
            detections.append({
                'start': match.start(),
                'end': match.end(),
                'text': match.group(),
                'label': 'PHONE',
                'confidence': 0.9
            })
        
        # Email patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            detections.append({
                'start': match.start(),
                'end': match.end(),
                'text': match.group(),
                'label': 'EMAIL',
                'confidence': 0.95
            })
        
        # SSN patterns
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        for match in re.finditer(ssn_pattern, text):
            detections.append({
                'start': match.start(),
                'end': match.end(),
                'text': match.group(),
                'label': 'SSN',
                'confidence': 0.98
            })
        
        # Date patterns
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{1,2}-\d{1,2}\b'
        for match in re.finditer(date_pattern, text):
            detections.append({
                'start': match.start(),
                'end': match.end(),
                'text': match.group(),
                'label': 'DATE',
                'confidence': 0.8
            })
        
        # Simple name patterns (capitalized words)
        name_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        for match in re.finditer(name_pattern, text):
            # Skip if it's a medical term or common word
            if match.group().lower() not in ['emergency room', 'intensive care', 'medical center']:
                detections.append({
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group(),
                    'label': 'PERSON',
                    'confidence': 0.6
                })
        
        return detections
    
    def evaluate_detector(self, test_data: pd.DataFrame) -> Dict:
        """Evaluate PHI detector performance"""
        logger.info("📊 Evaluating rule-based PHI detector...")
        
        total_true = 0
        total_detected = 0
        correct_detections = 0
        
        for _, row in test_data.iterrows():
            text = row['text']
            true_annotations = json.loads(row['annotations'])
            detected = self.rule_based_phi_detector(text)
            
            total_true += len(true_annotations)
            total_detected += len(detected)
            
            # Simple overlap-based evaluation
            for det in detected:
                for true_ann in true_annotations:
                    if (det['start'] <= true_ann['end'] and det['end'] >= true_ann['start'] 
                        and det['label'] == true_ann['label']):
                        correct_detections += 1
                        break
        
        precision = correct_detections / total_detected if total_detected > 0 else 0
        recall = correct_detections / total_true if total_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_true': total_true,
            'total_detected': total_detected,
            'correct_detections': correct_detections
        }
        
        logger.info(f"📈 Results: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        return results

def main():
    """Run emergency PHI bootstrap"""
    logger.info("🚨 EMERGENCY PHI BOOTSTRAP FOR MEDASSIST AI")
    logger.info("=" * 50)
    
    # Initialize bootstrap system
    bootstrap = EmergencyPHIBootstrap()
    
    # Create synthetic training data
    logger.info("🔥 Phase 1: Creating synthetic PHI training data...")
    train_data = bootstrap.create_training_dataset(num_samples=1000)
    
    # Save training data
    train_path = "data/processed/synthetic_phi_train.csv"
    train_data.to_csv(train_path, index=False)
    logger.info(f"💾 Saved training data: {train_path}")
    
    # Create test data
    logger.info("🧪 Phase 2: Creating test dataset...")
    test_data = bootstrap.create_training_dataset(num_samples=100)
    test_path = "data/processed/synthetic_phi_test.csv"
    test_data.to_csv(test_path, index=False)
    logger.info(f"💾 Saved test data: {test_path}")
    
    # Evaluate rule-based detector
    logger.info("📊 Phase 3: Evaluating rule-based PHI detector...")
    results = bootstrap.evaluate_detector(test_data)
    
    # Save results
    results_path = "data/processed/phi_bootstrap_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"📋 Saved evaluation results: {results_path}")
    
    # Summary
    logger.info("\n🎯 EMERGENCY PHI BOOTSTRAP COMPLETE!")
    logger.info("=" * 50)
    logger.info(f"✅ Training samples: {len(train_data)}")
    logger.info(f"✅ Test samples: {len(test_data)}")
    logger.info(f"✅ Rule-based F1 score: {results['f1_score']:.3f}")
    logger.info("\n📋 Next Steps:")
    logger.info("1. Train ClinicalBERT on synthetic data")
    logger.info("2. Quantize model for CPU deployment")
    logger.info("3. Integrate with Mojo preprocessing")
    logger.info("4. Submit real clinical dataset applications")
    logger.info("5. Fine-tune when real data becomes available")

if __name__ == "__main__":
    main()
