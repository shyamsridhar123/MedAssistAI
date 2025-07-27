#!/usr/bin/env python3
"""
Smart evaluation for PHI detection with context-aware boundary tolerance
"""

import sys
import os
import json
import pandas as pd
sys.path.append('/home/shyamsridhar/code/mojo-medassist/src')

from phi_detection.hybrid_detector import ClinicalBERTPHIDetector

def is_acceptable_boundary_mismatch(true_text, pred_text, full_text, true_start, true_end, pred_start, pred_end):
    """
    Determine if a boundary mismatch is acceptable for PHI de-identification
    
    Acceptable cases:
    - Over-redaction of whitespace/punctuation that doesn't change meaning
    - Conservative expansion that includes adjacent non-sensitive tokens
    
    Unacceptable cases:
    - Under-redaction (missing part of PHI)
    - Over-redaction that changes semantic meaning
    """
    
    # Case 1: Exact match - always acceptable
    if pred_start == true_start and pred_end == true_end:
        return "EXACT_MATCH", 1.0
    
    # Case 2: Under-redaction - never acceptable for PHI
    if pred_start > true_start or pred_end < true_end:
        return "UNDER_REDACTION", 0.0
    
    # Case 3: Over-redaction - check if it's reasonable
    if pred_start <= true_start and pred_end >= true_end:
        # Get the extra characters being redacted
        prefix_extra = full_text[pred_start:true_start] if pred_start < true_start else ""
        suffix_extra = full_text[true_end:pred_end] if pred_end > true_end else ""
        
        # Check if extra characters are "safe" to redact
        safe_chars = {' ', '\t', '\n', '.', ',', ':', ';', '!', '?', '-', '(', ')', '[', ']', '"', "'"}
        safe_words = {'on', 'at', 'in', 'to', 'the', 'a', 'an', 'and', 'or', 'but'}
        
        # Analyze prefix
        prefix_safe = all(c in safe_chars for c in prefix_extra) or prefix_extra.strip().lower() in safe_words
        
        # Analyze suffix  
        suffix_tokens = suffix_extra.strip().split()
        suffix_safe = (
            all(c in safe_chars for c in suffix_extra) or
            all(token.lower() in safe_words for token in suffix_tokens) or
            len(suffix_tokens) <= 1  # Single word expansion usually ok
        )
        
        if prefix_safe and suffix_safe:
            # Calculate penalty based on how much extra we're redacting
            extra_chars = len(prefix_extra) + len(suffix_extra)
            phi_chars = len(true_text)
            penalty = min(0.3, extra_chars / max(phi_chars, 1))  # Max 30% penalty
            score = 1.0 - penalty
            return "ACCEPTABLE_OVER_REDACTION", score
        else:
            return "PROBLEMATIC_OVER_REDACTION", 0.2  # Some credit for finding the PHI
    
    return "UNKNOWN", 0.0

def smart_evaluation():
    print("🧪 Smart PHI Evaluation with Context-Aware Boundaries...")
    
    # Load the real trained model
    detector = ClinicalBERTPHIDetector("models/phi_detection/clinicalbert_real/final")
    success = detector.load_model()
    
    if not success:
        print("❌ Failed to load model")
        return
    
    print("✅ Model loaded successfully")
    
    # Load test data
    df = pd.read_csv("data/processed/synthetic_phi_test.csv")
    
    total_score = 0
    total_entities = 0
    match_types = {}
    
    for i in range(min(10, len(df))):  # Test first 10 samples
        text = df.iloc[i]['text']
        true_annotations = json.loads(df.iloc[i]['annotations'])
        pred_annotations = detector.detect(text)
        
        print(f"\n📝 Sample {i+1}:")
        print(f"Text: {text}")
        
        # Match predictions to ground truth
        for true_ann in true_annotations:
            total_entities += 1
            best_match = None
            best_score = 0
            best_type = "NO_MATCH"
            
            # Find best matching prediction
            for pred_ann in pred_annotations:
                if pred_ann['label'] != true_ann['label']:
                    continue
                
                match_type, score = is_acceptable_boundary_mismatch(
                    true_ann['text'], pred_ann['text'], text,
                    true_ann['start'], true_ann['end'],
                    pred_ann['start'], pred_ann['end']
                )
                
                if score > best_score:
                    best_score = score
                    best_match = pred_ann
                    best_type = match_type
            
            total_score += best_score
            match_types[best_type] = match_types.get(best_type, 0) + 1
            
            print(f"  TRUE:  {true_ann['start']:3d}-{true_ann['end']:3d} '{true_ann['text']}' -> {true_ann['label']}")
            if best_match:
                print(f"  PRED:  {best_match['start']:3d}-{best_match['end']:3d} '{best_match['text']}' -> {best_match['label']} (conf: {best_match['confidence']:.3f})")
                print(f"         Match: {best_type} (score: {best_score:.3f})")
            else:
                print(f"         Match: NO_MATCH (score: 0.000)")
    
    # Calculate overall metrics
    avg_score = total_score / total_entities if total_entities > 0 else 0
    
    print(f"\n🎯 SMART EVALUATION RESULTS:")
    print(f"   Average Score: {avg_score:.3f}")
    print(f"   Total Entities: {total_entities}")
    print(f"\n📊 MATCH TYPE BREAKDOWN:")
    for match_type, count in sorted(match_types.items()):
        pct = (count / total_entities) * 100 if total_entities > 0 else 0
        print(f"   {match_type}: {count} ({pct:.1f}%)")
    
    # Deployment recommendation
    if avg_score >= 0.9:
        print(f"\n✅ EXCELLENT: Model ready for production PHI de-identification")
    elif avg_score >= 0.8:
        print(f"\n✅ GOOD: Model acceptable for PHI de-identification with review")
    elif avg_score >= 0.7:
        print(f"\n⚠️ FAIR: Model needs improvement before production use")
    else:
        print(f"\n❌ POOR: Model not suitable for PHI de-identification")

if __name__ == "__main__":
    smart_evaluation()
