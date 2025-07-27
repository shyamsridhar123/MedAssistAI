#!/usr/bin/env python3
"""
Comprehensive Model Evaluation for PHI Detection
MedAssist AI - Option C Implementation

Evaluates rule-based, ClinicalBERT, and hybrid detection systems.
"""

import json
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import sys
import os

# Add project root to path
sys.path.append('/home/shyamsridhar/code/mojo-medassist')

from src.phi_detection.hybrid_detector import (
    RuleBasedPHIDetector, 
    ClinicalBERTPHIDetector, 
    HybridPHIDetector
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PHIEvaluator:
    """Comprehensive evaluation of PHI detection systems"""
    
    def __init__(self):
        self.rule_detector = RuleBasedPHIDetector()
        self.bert_detector = ClinicalBERTPHIDetector("models/phi_detection/clinicalbert_real/final")  # Use real trained model
        self.hybrid_detector = HybridPHIDetector()      # Use default path
        
        # PHI entity types
        self.entity_types = ['PERSON', 'LOCATION', 'DATE', 'PHONE', 'SSN', 'EMAIL', 'MRN']
    
    def load_test_data(self, test_path: str) -> Tuple[List[str], List[List[Dict]]]:
        """Load test data for evaluation"""
        logger.info(f"📊 Loading test data from {test_path}")
        
        df = pd.read_csv(test_path)
        texts = df['text'].tolist()
        annotations = [json.loads(ann) for ann in df['annotations']]
        
        logger.info(f"✅ Loaded {len(texts)} test samples")
        return texts, annotations
    
    def evaluate_detector(self, detector, texts: List[str], 
                         true_annotations: List[List[Dict]], 
                         detector_name: str) -> Dict:
        """Evaluate a single detector"""
        logger.info(f"🔍 Evaluating {detector_name} detector...")
        
        all_true_entities = []
        all_pred_entities = []
        
        total_samples = len(texts)
        
        for i, (text, true_anns) in enumerate(zip(texts, true_annotations)):
            if (i + 1) % 20 == 0:
                logger.info(f"  Processed {i + 1}/{total_samples} samples...")
            
            # Get predictions
            pred_anns = detector.detect(text)
            
            # Convert to evaluation format
            true_entities = self._annotations_to_entities(true_anns)
            pred_entities = self._annotations_to_entities(pred_anns)
            
            all_true_entities.extend(true_entities)
            all_pred_entities.extend(pred_entities)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_true_entities, all_pred_entities)
        metrics['detector_name'] = detector_name
        
        logger.info(f"📈 {detector_name} Results:")
        logger.info(f"  Precision: {metrics['precision']:.3f}")
        logger.info(f"  Recall: {metrics['recall']:.3f}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.3f}")
        
        return metrics
    
    def _annotations_to_entities(self, annotations: List[Dict]) -> List[Tuple]:
        """Convert annotations to entity tuples for evaluation"""
        entities = []
        for ann in annotations:
            entities.append((ann['start'], ann['end'], ann['label']))
        return entities
    
    def _calculate_metrics(self, true_entities: List[Tuple], 
                          pred_entities: List[Tuple]) -> Dict:
        """Calculate precision, recall, F1 for entity detection with overlap matching"""
        
        # Match entities with overlap tolerance
        matched_true = set()
        matched_pred = set()
        
        for i, pred_entity in enumerate(pred_entities):
            pred_start, pred_end, pred_label = pred_entity
            best_match = None
            best_overlap = 0
            
            for j, true_entity in enumerate(true_entities):
                if j in matched_true:
                    continue
                    
                true_start, true_end, true_label = true_entity
                
                # Only match same label types
                if pred_label != true_label:
                    continue
                
                # Calculate overlap
                overlap_start = max(pred_start, true_start)
                overlap_end = min(pred_end, true_end)
                
                if overlap_start < overlap_end:  # There is overlap
                    overlap_len = overlap_end - overlap_start
                    true_len = true_end - true_start
                    pred_len = pred_end - pred_start
                    
                    # IoU-style overlap score
                    union_len = max(true_end, pred_end) - min(true_start, pred_start)
                    overlap_score = overlap_len / union_len
                    
                    # Require at least 50% overlap for a match
                    if overlap_score > 0.5 and overlap_score > best_overlap:
                        best_overlap = overlap_score
                        best_match = j
            
            # If we found a good match, mark both as matched
            if best_match is not None:
                matched_true.add(best_match)
                matched_pred.add(i)
        
        # Calculate metrics
        true_positives = len(matched_pred)
        false_positives = len(pred_entities) - len(matched_pred)
        false_negatives = len(true_entities) - len(matched_true)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Per-entity-type metrics (simplified for now)
        entity_metrics = {}
        for entity_type in self.entity_types:
            true_type_count = sum(1 for e in true_entities if e[2] == entity_type)
            pred_type_count = sum(1 for e in pred_entities if e[2] == entity_type)
            
            entity_metrics[entity_type] = {
                'precision': 0.0,  # Would need more complex calculation
                'recall': 0.0,
                'f1_score': 0.0,
                'true_count': true_type_count,
                'pred_count': pred_type_count
            }
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'entity_metrics': entity_metrics
        }
    
    def compare_detectors(self, test_path: str) -> Dict:
        """Compare all detection methods"""
        logger.info("🏆 COMPREHENSIVE PHI DETECTOR COMPARISON")
        logger.info("=" * 50)
        
        # Load test data
        texts, true_annotations = self.load_test_data(test_path)
        
        # Initialize hybrid detector
        self.hybrid_detector.initialize()
        
        # Evaluate each detector
        results = {}
        
        # 1. Rule-based detector
        results['rule_based'] = self.evaluate_detector(
            self.rule_detector, texts, true_annotations, "Rule-Based"
        )
        
        # 2. ClinicalBERT detector (if available)
        if self.hybrid_detector.bert_available:
            # Load the individual bert detector too
            self.bert_detector.load_model()
            results['clinical_bert'] = self.evaluate_detector(
                self.bert_detector, texts, true_annotations, "ClinicalBERT"
            )
        else:
            logger.warning("⚠️ ClinicalBERT not available, skipping evaluation")
        
        # 3. Hybrid detector
        results['hybrid'] = self.evaluate_detector(
            self.hybrid_detector, texts, true_annotations, "Hybrid"
        )
        
        # Generate comparison report
        comparison_report = self._generate_comparison_report(results)
        
        return {
            'individual_results': results,
            'comparison_report': comparison_report
        }
    
    def _generate_comparison_report(self, results: Dict) -> Dict:
        """Generate comprehensive comparison report"""
        logger.info("📊 Generating comparison report...")
        
        # Overall performance comparison
        performance_summary = {}
        for method, metrics in results.items():
            performance_summary[method] = {
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            }
        
        # Find best performing method
        best_f1_method = max(performance_summary.items(), key=lambda x: x[1]['f1_score'])
        best_precision_method = max(performance_summary.items(), key=lambda x: x[1]['precision'])
        best_recall_method = max(performance_summary.items(), key=lambda x: x[1]['recall'])
        
        # Entity-type performance analysis
        entity_analysis = {}
        for entity_type in self.entity_types:
            entity_analysis[entity_type] = {}
            for method, metrics in results.items():
                if entity_type in metrics['entity_metrics']:
                    entity_analysis[entity_type][method] = metrics['entity_metrics'][entity_type]['f1_score']
        
        report = {
            'performance_summary': performance_summary,
            'best_performers': {
                'f1_score': best_f1_method,
                'precision': best_precision_method,
                'recall': best_recall_method
            },
            'entity_type_analysis': entity_analysis,
            'recommendations': self._generate_recommendations(results)
        }
        
        return report
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate deployment recommendations based on results"""
        recommendations = []
        
        # Check if hybrid is best overall
        if 'hybrid' in results and results['hybrid']['f1_score'] == max(r['f1_score'] for r in results.values()):
            recommendations.append("✅ Deploy hybrid system for optimal balanced performance")
        
        # Check for high precision needs
        rule_precision = results.get('rule_based', {}).get('precision', 0)
        if rule_precision > 0.9:
            recommendations.append("🎯 Rule-based detector excellent for high-precision scenarios")
        
        # Check for high recall needs
        hybrid_recall = results.get('hybrid', {}).get('recall', 0)
        if hybrid_recall > 0.85:
            recommendations.append("🔍 Hybrid system recommended for comprehensive PHI coverage")
        
        # Performance gaps
        best_f1 = max(r['f1_score'] for r in results.values())
        if best_f1 < 0.9:
            recommendations.append("⚠️ Consider real clinical data training to reach 90%+ F1 score")
        
        return recommendations
    
    def save_results(self, results: Dict, output_path: str = "models/phi_detection/evaluation_results.json"):
        """Save evaluation results"""
        logger.info(f"💾 Saving evaluation results to {output_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("✅ Results saved successfully")
    
    def create_performance_visualization(self, results: Dict, 
                                       output_path: str = "models/phi_detection/performance_comparison.png"):
        """Create performance visualization"""
        logger.info("📈 Creating performance visualization...")
        
        try:
            # Prepare data for plotting
            methods = []
            precision_scores = []
            recall_scores = []
            f1_scores = []
            
            for method, metrics in results['individual_results'].items():
                methods.append(method.replace('_', ' ').title())
                precision_scores.append(metrics['precision'])
                recall_scores.append(metrics['recall'])
                f1_scores.append(metrics['f1_score'])
            
            # Create subplot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Performance comparison bar chart
            x = np.arange(len(methods))
            width = 0.25
            
            ax1.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
            ax1.bar(x, recall_scores, width, label='Recall', alpha=0.8)
            ax1.bar(x + width, f1_scores, width, label='F1 Score', alpha=0.8)
            
            ax1.set_xlabel('Detection Method')
            ax1.set_ylabel('Score')
            ax1.set_title('PHI Detection Performance Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(methods, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Entity type performance heatmap
            entity_data = []
            entity_methods = []
            
            for method in results['individual_results'].keys():
                if method in results['individual_results']:
                    method_scores = []
                    for entity_type in self.entity_types:
                        score = results['individual_results'][method]['entity_metrics'].get(
                            entity_type, {}
                        ).get('f1_score', 0)
                        method_scores.append(score)
                    entity_data.append(method_scores)
                    entity_methods.append(method.replace('_', ' ').title())
            
            if entity_data:
                sns.heatmap(entity_data, 
                           xticklabels=self.entity_types,
                           yticklabels=entity_methods,
                           annot=True, 
                           fmt='.2f',
                           cmap='YlOrRd',
                           ax=ax2)
                ax2.set_title('F1 Score by Entity Type')
                ax2.set_xlabel('PHI Entity Type')
                ax2.set_ylabel('Detection Method')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Visualization saved to {output_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Visualization creation failed: {e}")

def main():
    """Main evaluation pipeline"""
    logger.info("🚨 PHI DETECTION COMPREHENSIVE EVALUATION")
    logger.info("=" * 50)
    
    # Initialize evaluator
    evaluator = PHIEvaluator()
    
    # Run comprehensive comparison
    results = evaluator.compare_detectors('data/processed/synthetic_phi_test.csv')
    
    # Save results
    evaluator.save_results(results)
    
    # Create visualization
    evaluator.create_performance_visualization(results)
    
    # Print final recommendations
    logger.info("\n🎯 DEPLOYMENT RECOMMENDATIONS:")
    for rec in results['comparison_report']['recommendations']:
        logger.info(f"  {rec}")
    
    # Print summary
    logger.info("\n📊 PERFORMANCE SUMMARY:")
    for method, metrics in results['comparison_report']['performance_summary'].items():
        logger.info(f"  {method.title()}: P={metrics['precision']:.3f}, "
                   f"R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
    
    logger.info("\n🏆 Evaluation complete! Check models/phi_detection/ for detailed results.")

if __name__ == "__main__":
    main()
