#!/usr/bin/env python3
"""
ClinicalBERT Training for PHI Detection
MedAssist AI - Hybrid PHI Detection System

This script downloads and fine-tunes ClinicalBERT on synthetic PHI data.
Part of Option C (Balanced Approach) implementation.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from sklearn.metrics import classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PHIDataset(Dataset):
    """Dataset for PHI detection training"""
    
    def __init__(self, texts: List[str], annotations: List[List[Dict]], 
                 tokenizer, max_length: int = 512):
        self.texts = texts
        self.annotations = annotations
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # PHI label mapping
        self.label2id = {
            'O': 0,
            'B-PERSON': 1, 'I-PERSON': 2,
            'B-LOCATION': 3, 'I-LOCATION': 4,
            'B-DATE': 5, 'I-DATE': 6,
            'B-PHONE': 7, 'I-PHONE': 8,
            'B-SSN': 9, 'I-SSN': 10,
            'B-EMAIL': 11, 'I-EMAIL': 12,
            'B-MRN': 13, 'I-MRN': 14
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        annotations = self.annotations[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # Create labels
        labels = self.create_labels(text, annotations, encoding['offset_mapping'][0])
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def create_labels(self, text: str, annotations: List[Dict], offset_mapping) -> List[int]:
        """Create BIO labels for tokenized text"""
        labels = [0] * len(offset_mapping)  # Initialize with 'O'
        
        for annotation in annotations:
            start_char = annotation['start']
            end_char = annotation['end']
            entity_type = annotation['label']
            
            # Find tokens that overlap with the entity
            entity_tokens = []
            for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                if token_start is None or token_end is None:
                    continue
                    
                # Check if token overlaps with entity
                if token_start < end_char and token_end > start_char:
                    entity_tokens.append(token_idx)
            
            # Apply BIO labeling
            for i, token_idx in enumerate(entity_tokens):
                if i == 0:  # First token gets B- label
                    labels[token_idx] = self.label2id[f'B-{entity_type}']
                else:  # Subsequent tokens get I- label
                    labels[token_idx] = self.label2id[f'I-{entity_type}']
        
        return labels

class ClinicalBERTPHITrainer:
    """ClinicalBERT training pipeline for PHI detection"""
    
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label2id = {
            'O': 0,
            'B-PERSON': 1, 'I-PERSON': 2,
            'B-LOCATION': 3, 'I-LOCATION': 4,
            'B-DATE': 5, 'I-DATE': 6,
            'B-PHONE': 7, 'I-PHONE': 8,
            'B-SSN': 9, 'I-SSN': 10,
            'B-EMAIL': 11, 'I-EMAIL': 12,
            'B-MRN': 13, 'I-MRN': 14
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
    def setup_model(self):
        """Download and setup ClinicalBERT for token classification"""
        logger.info(f"🔽 Downloading ClinicalBERT: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id
            )
            logger.info("✅ ClinicalBERT setup complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to download ClinicalBERT: {e}")
            logger.info("🔄 Falling back to standard BERT...")
            self.model_name = "bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id
            )
            logger.info("✅ Standard BERT setup complete")
    
    def load_data(self, train_path: str, test_path: str) -> Tuple[PHIDataset, PHIDataset]:
        """Load and preprocess training data"""
        logger.info("📊 Loading synthetic PHI data...")
        
        # Load training data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Parse annotations
        train_texts = train_df['text'].tolist()
        train_annotations = [json.loads(ann) for ann in train_df['annotations']]
        
        test_texts = test_df['text'].tolist()
        test_annotations = [json.loads(ann) for ann in test_df['annotations']]
        
        # Create datasets
        train_dataset = PHIDataset(train_texts, train_annotations, self.tokenizer)
        test_dataset = PHIDataset(test_texts, test_annotations, self.tokenizer)
        
        logger.info(f"✅ Loaded {len(train_dataset)} training samples, {len(test_dataset)} test samples")
        return train_dataset, test_dataset
    
    def train(self, train_dataset: PHIDataset, test_dataset: PHIDataset, 
              output_dir: str = "models/phi_detection/clinicalbert"):
        """Train ClinicalBERT on PHI detection task"""
        logger.info("🚀 Starting ClinicalBERT training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            eval_strategy="epoch",  # Fixed: was evaluation_strategy
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None  # Disable wandb
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train model
        logger.info("🔥 Training in progress...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        logger.info(f"✅ Model saved to {output_dir}")
        
        return trainer
    
    def evaluate(self, trainer, test_dataset: PHIDataset) -> Dict:
        """Evaluate trained model performance"""
        logger.info("📊 Evaluating ClinicalBERT performance...")
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        predictions_np = np.argmax(predictions.predictions, axis=2)
        
        # Flatten labels and predictions
        true_labels = []
        pred_labels = []
        
        for i in range(len(test_dataset)):
            item = test_dataset[i]
            true_seq = item['labels'].numpy()
            pred_seq = predictions_np[i]
            attention_mask = item['attention_mask'].numpy()
            
            # Only consider non-padded tokens
            for j in range(len(true_seq)):
                if attention_mask[j] == 1 and true_seq[j] != -100:
                    true_labels.append(true_seq[j])
                    pred_labels.append(pred_seq[j])
        
        # Calculate metrics
        f1 = f1_score(true_labels, pred_labels, average='weighted')
        
        # Detailed classification report
        target_names = list(self.id2label.values())
        report = classification_report(
            true_labels, pred_labels, 
            target_names=target_names, 
            output_dict=True, 
            zero_division=0
        )
        
        results = {
            'f1_score': f1,
            'classification_report': report
        }
        
        logger.info(f"📈 ClinicalBERT F1 Score: {f1:.3f}")
        return results
    
    def quantize_model(self, model_path: str, output_path: str):
        """Quantize model for CPU deployment"""
        logger.info("⚡ Quantizing model for CPU deployment...")
        
        try:
            import torch.quantization as quantization
            
            # Load model
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            model.eval()
            
            # Apply dynamic quantization
            quantized_model = quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            
            # Save quantized model
            torch.save(quantized_model.state_dict(), f"{output_path}/quantized_model.pth")
            logger.info(f"✅ Quantized model saved to {output_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Quantization failed: {e}")
            logger.info("📦 Saving regular model instead...")

def main():
    """Main training pipeline"""
    logger.info("🚨 CLINICALBERT PHI DETECTION TRAINING")
    logger.info("=" * 50)
    
    # Initialize trainer
    trainer = ClinicalBERTPHITrainer()
    
    # Setup model
    trainer.setup_model()
    
    # Load data
    train_dataset, test_dataset = trainer.load_data(
        'data/processed/synthetic_phi_train.csv',
        'data/processed/synthetic_phi_test.csv'
    )
    
    # Train model
    model_trainer = trainer.train(train_dataset, test_dataset)
    
    # Evaluate performance
    results = trainer.evaluate(model_trainer, test_dataset)
    
    # Save results
    with open('models/phi_detection/clinicalbert/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Quantize for deployment
    trainer.quantize_model(
        'models/phi_detection/clinicalbert',
        'models/phi_detection/clinicalbert/quantized'
    )
    
    logger.info("🎯 ClinicalBERT training complete!")
    logger.info(f"📊 Final F1 Score: {results['f1_score']:.3f}")
    logger.info("📋 Next: Implement hybrid detection system")

if __name__ == "__main__":
    main()
