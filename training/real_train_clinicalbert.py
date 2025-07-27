#!/usr/bin/env python3
"""
REAL ClinicalBERT Training - With Proper Labels This Time
No more bullshit zero labels
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, classification_report

# Force CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealPHIDataset(Dataset):
    """Dataset with ACTUAL PHI labels"""
    
    def __init__(self, texts, annotations, tokenizer, max_length=128):
        self.texts = texts
        self.annotations = annotations
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Real label mapping
        self.label2id = {
            'O': 0,           # No PHI
            'B-PERSON': 1,    'I-PERSON': 2,
            'B-LOCATION': 3,  'I-LOCATION': 4,
            'B-DATE': 5,      'I-DATE': 6,
            'B-PHONE': 7,     'I-PHONE': 8,
            'B-SSN': 9,       'I-SSN': 10,
            'B-EMAIL': 11,    'I-EMAIL': 12,
            'B-MRN': 13,      'I-MRN': 14
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        annotations = json.loads(self.annotations[idx]) if isinstance(self.annotations[idx], str) else self.annotations[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # Create labels array - start with all O (no PHI)
        labels = torch.zeros(self.max_length, dtype=torch.long)
        
        # Get token offsets
        offset_mapping = encoding['offset_mapping'][0]
        
        # Map annotations to tokens
        for ann in annotations:
            start_char = ann['start']
            end_char = ann['end']
            entity_label = ann['label']
            
            # Find tokens that overlap with this annotation
            token_start_idx = None
            token_end_idx = None
            
            for i, (token_start, token_end) in enumerate(offset_mapping):
                # Skip special tokens
                if token_start == 0 and token_end == 0:
                    continue
                    
                # Check if token overlaps with annotation
                if token_start < end_char and token_end > start_char:
                    if token_start_idx is None:
                        token_start_idx = i
                    token_end_idx = i
            
            # Assign BIO labels
            if token_start_idx is not None and token_end_idx is not None:
                for i in range(token_start_idx, token_end_idx + 1):
                    if i == token_start_idx:
                        # First token gets B- label
                        b_label = f'B-{entity_label}'
                        if b_label in self.label2id:
                            labels[i] = self.label2id[b_label]
                    else:
                        # Subsequent tokens get I- label
                        i_label = f'I-{entity_label}'
                        if i_label in self.label2id:
                            labels[i] = self.label2id[i_label]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

def main():
    """Train ClinicalBERT with REAL labels"""
    logger.info("🚨 REAL CLINICALBERT TRAINING (WITH PROPER LABELS)")
    logger.info("=" * 60)
    
    # Ensure directories exist
    os.makedirs('models/phi_detection/clinicalbert_real', exist_ok=True)
    
    # Load data
    logger.info("📊 Loading synthetic PHI data...")
    train_df = pd.read_csv('data/processed/synthetic_phi_train.csv')
    test_df = pd.read_csv('data/processed/synthetic_phi_test.csv')
    
    # Setup model
    logger.info("🔽 Loading ClinicalBERT...")
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=15)
    
    # Force CPU
    device = torch.device('cpu')
    model.to(device)
    
    logger.info("✅ Model loaded on CPU")
    
    # Create datasets with REAL labels
    train_dataset = RealPHIDataset(
        train_df['text'].tolist(),
        train_df['annotations'].tolist(),
        tokenizer
    )
    
    test_dataset = RealPHIDataset(
        test_df['text'].tolist(), 
        test_df['annotations'].tolist(),
        tokenizer
    )
    
    logger.info(f"✅ Loaded {len(train_dataset)} train, {len(test_dataset)} test samples")
    
    # Check a sample to make sure labels are right
    sample = train_dataset[0]
    non_zero_labels = (sample['labels'] != 0).sum().item()
    logger.info(f"🔍 Sample check: {non_zero_labels} PHI tokens in first sample")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='models/phi_detection/clinicalbert_real',
        num_train_epochs=3,  # More epochs for real training
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=25,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        use_cpu=True,
        dataloader_num_workers=0,
        remove_unused_columns=False
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("🚀 Starting REAL training with proper labels...")
    trainer.train()
    
    # Save
    trainer.save_model('models/phi_detection/clinicalbert_real/final')
    tokenizer.save_pretrained('models/phi_detection/clinicalbert_real/final')
    
    logger.info("✅ REAL training complete!")
    logger.info("📁 Model saved to models/phi_detection/clinicalbert_real/final/")
    
    # Quick test
    logger.info("🧪 Quick test on trained model...")
    test_text = "Patient John Smith was born on 12/25/1980"
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_classes = predictions.argmax(-1)
    
    phi_predictions = (predicted_classes != 0).sum().item()
    logger.info(f"   Test result: {phi_predictions} PHI tokens detected in test sentence")
    
    if phi_predictions > 0:
        logger.info("🎯 SUCCESS! Model is now detecting PHI!")
    else:
        logger.info("⚠️ Still not detecting PHI - may need more training")
    
    # Save results
    with open('models/phi_detection/clinicalbert_real/training_results.json', 'w') as f:
        json.dump({
            'status': 'real_training_complete',
            'model_name': model_name,
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset),
            'labels_used': 15,
            'test_phi_detections': phi_predictions
        }, f, indent=2)
    
    logger.info("🎯 REAL ClinicalBERT training complete!")
    return 0

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Real training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
