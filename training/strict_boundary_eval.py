#!/usr/bin/env python3
"""
Test strict vs lenient boundary evaluation for PHI detection
"""

import sys
import os
import json
import pandas as pd
sys.path.append('/home/shyamsridhar/code/mojo-medassist/src')

from phi_detection.hybrid_detector import ClinicalBERTPHIDetector

def evaluate_strict_boundaries():
    print("🎯 Evaluating with STRICT boundary matching for PHI de-identification...")
    
    # Load the real trained model
    detector = ClinicalBERTPHIDetector("models/phi_detection/clinicalbert_real/final")
    success = detector.load_model()
    
    if not success:
        print("❌ Failed to load model")
        return
    
    print("✅ Model loaded successfully")
    
    # Load test data
    df = pd.read_csv("data/processed/synthetic_phi_test.csv")
    
    total_true = 0
    total_pred = 0
    exact_matches = 0
    partial_matches = 0
    
    print("\n🔍 Analyzing boundary precision...")
    
    for i in range(min(10, len(df))):  # Test first 10 samples
        text = df.iloc[i]['text']
        true_annotations = json.loads(df.iloc[i]['annotations'])
        pred_annotations = detector.detect(text)
        
        total_true += len(true_annotations)
        total_pred += len(pred_annotations)
        
        # Check for exact and partial matches
        for true_ann in true_annotations:
            true_start, true_end, true_label = true_ann['start'], true_ann['end'], true_ann['label']
            
            exact_match = False
            partial_match = False
            
            for pred_ann in pred_annotations:
                pred_start, pred_end, pred_label = pred_ann['start'], pred_ann['end'], pred_ann['label']
                
                if pred_label == true_label:
                    # Exact match
                    if pred_start == true_start and pred_end == true_end:
                        exact_match = True
                        break
                    # Partial overlap
                    elif (pred_start < true_end and pred_end > true_start):
                        partial_match = True
                        
                        # Show the difference
                        if i < 3:  # Only show details for first 3
                            true_text = text[true_start:true_end]
                            pred_text = text[pred_start:pred_end]
                            print(f"  ⚠️  Boundary mismatch ({true_label}):")
                            print(f"      TRUE: [{true_start:2d}-{true_end:2d}] '{true_text}'")
                            print(f"      PRED: [{pred_start:2d}-{pred_end:2d}] '{pred_text}'")
                            print(f"      IMPACT: Would redact '{pred_text}' instead of '{true_text}'")
            
            if exact_match:
                exact_matches += 1
            elif partial_match:
                partial_matches += 1
    
    # Calculate strict metrics
    strict_precision = exact_matches / total_pred if total_pred > 0 else 0
    strict_recall = exact_matches / total_true if total_true > 0 else 0
    strict_f1 = 2 * strict_precision * strict_recall / (strict_precision + strict_recall) if (strict_precision + strict_recall) > 0 else 0
    
    # Calculate lenient metrics (exact + partial)
    lenient_matches = exact_matches + partial_matches
    lenient_precision = lenient_matches / total_pred if total_pred > 0 else 0
    lenient_recall = lenient_matches / total_true if total_true > 0 else 0
    lenient_f1 = 2 * lenient_precision * lenient_recall / (lenient_precision + lenient_recall) if (lenient_precision + lenient_recall) > 0 else 0
    
    print(f"\n📊 BOUNDARY EVALUATION RESULTS:")
    print(f"   Total True Entities: {total_true}")
    print(f"   Total Pred Entities: {total_pred}")
    print(f"   Exact Matches: {exact_matches}")
    print(f"   Partial Matches: {partial_matches}")
    print(f"")
    print(f"📏 STRICT BOUNDARIES (Production PHI De-ID):")
    print(f"   Precision: {strict_precision:.3f}")
    print(f"   Recall: {strict_recall:.3f}")
    print(f"   F1 Score: {strict_f1:.3f}")
    print(f"")
    print(f"🤝 LENIENT BOUNDARIES (Research/Development):")
    print(f"   Precision: {lenient_precision:.3f}")
    print(f"   Recall: {lenient_recall:.3f}")
    print(f"   F1 Score: {lenient_f1:.3f}")
    print(f"")
    print(f"💡 RECOMMENDATION FOR PHI DE-IDENTIFICATION:")
    if strict_f1 > 0.90:
        print(f"   ✅ PRODUCTION READY - High strict boundary accuracy")
    elif strict_f1 > 0.75:
        print(f"   ⚠️  NEEDS IMPROVEMENT - Boundary precision issues")
    else:
        print(f"   ❌ NOT READY - Significant boundary problems")

if __name__ == "__main__":
    evaluate_strict_boundaries()
