#!/usr/bin/env python3
"""
validate_datasets.py - Validate downloaded datasets for MedAssist AI

This script checks the integrity, format, and content of downloaded datasets
to ensure they're suitable for training the de-identification, summarization,
and diagnostic models.

Usage: 
    python3 validate_datasets.py [data_directory]
    magic run python3 validate_datasets.py [data_directory]
"""

import os
import sys
import json
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dataset_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetValidator:
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw"
        self.validation_results = {}
        
    def validate_all_datasets(self) -> Dict:
        """Validate all available datasets"""
        logger.info("🔍 Starting comprehensive dataset validation...")
        
        # Validate each dataset type
        self.validate_synthea_data()
        self.validate_kaggle_data()
        self.validate_huggingface_data()
        self.validate_github_corpora()
        
        # Generate summary report
        self.generate_validation_report()
        
        return self.validation_results
    
    def validate_synthea_data(self):
        """Validate Synthea synthetic patient data"""
        logger.info("📊 Validating Synthea synthetic data...")
        
        synthea_dir = self.raw_dir / "synthea"
        result = {
            "status": "missing",
            "samples": {},
            "issues": [],
            "file_count": 0,
            "estimated_patients": 0
        }
        
        if not synthea_dir.exists():
            result["issues"].append("Synthea directory not found")
            self.validation_results["synthea"] = result
            return
        
        # Check 1K sample
        sample_1k = synthea_dir / "1k_sample"
        if sample_1k.exists():
            try:
                csv_files = list(sample_1k.glob("*.csv"))
                result["samples"]["1k"] = {
                    "files": len(csv_files),
                    "file_list": [f.name for f in csv_files[:5]]  # First 5 files
                }
                
                # Validate key files
                expected_files = ["person.csv", "encounters.csv", "conditions.csv"]
                for expected in expected_files:
                    file_path = sample_1k / expected
                    if file_path.exists():
                        try:
                            df = pd.read_csv(file_path, nrows=5)  # Read first 5 rows
                            result["samples"]["1k"][expected] = {
                                "rows_sample": len(df),
                                "columns": list(df.columns)
                            }
                        except Exception as e:
                            result["issues"].append(f"Error reading {expected}: {str(e)}")
                    else:
                        result["issues"].append(f"Missing expected file: {expected}")
                        
            except Exception as e:
                result["issues"].append(f"Error validating 1k sample: {str(e)}")
        
        # Check 100K sample
        sample_100k = synthea_dir / "100k_sample"
        if sample_100k.exists():
            try:
                csv_files = list(sample_100k.glob("*.csv"))
                result["samples"]["100k"] = {
                    "files": len(csv_files),
                    "file_list": [f.name for f in csv_files[:5]]
                }
                
                # Estimate patient count from person.csv if available
                person_file = sample_100k / "person.csv"
                if person_file.exists():
                    try:
                        person_df = pd.read_csv(person_file)
                        result["estimated_patients"] = len(person_df)
                    except Exception as e:
                        result["issues"].append(f"Could not count patients: {str(e)}")
                        
            except Exception as e:
                result["issues"].append(f"Error validating 100k sample: {str(e)}")
        
        # Determine overall status
        if result["samples"]:
            result["status"] = "valid" if not result["issues"] else "valid_with_issues"
        
        self.validation_results["synthea"] = result
        logger.info(f"✅ Synthea validation complete: {result['status']}")
    
    def validate_kaggle_data(self):
        """Validate Kaggle datasets"""
        logger.info("📊 Validating Kaggle datasets...")
        
        kaggle_dir = self.raw_dir / "kaggle"
        result = {
            "status": "missing",
            "datasets": {},
            "issues": []
        }
        
        if not kaggle_dir.exists():
            result["issues"].append("Kaggle directory not found")
            self.validation_results["kaggle"] = result
            return
        
        # Check healthcare dataset
        healthcare_dir = kaggle_dir / "healthcare-dataset"
        if healthcare_dir.exists():
            try:
                csv_files = list(healthcare_dir.glob("*.csv"))
                if csv_files:
                    # Read first CSV file found
                    df = pd.read_csv(csv_files[0], nrows=10)
                    result["datasets"]["healthcare"] = {
                        "files": len(csv_files),
                        "sample_rows": len(df),
                        "columns": list(df.columns),
                        "file_size_mb": round(csv_files[0].stat().st_size / 1024 / 1024, 2)
                    }
                else:
                    result["issues"].append("Healthcare dataset directory exists but no CSV files found")
            except Exception as e:
                result["issues"].append(f"Error validating healthcare dataset: {str(e)}")
        
        # Check medical text dataset
        medical_text_dir = kaggle_dir / "medical-text"
        if medical_text_dir.exists():
            try:
                text_files = list(medical_text_dir.glob("*.csv")) + list(medical_text_dir.glob("*.txt"))
                if text_files:
                    result["datasets"]["medical_text"] = {
                        "files": len(text_files),
                        "file_types": [f.suffix for f in text_files]
                    }
                    
                    # Try to read first file
                    try:
                        if text_files[0].suffix == '.csv':
                            df = pd.read_csv(text_files[0], nrows=5)
                            result["datasets"]["medical_text"]["sample_columns"] = list(df.columns)
                    except Exception as e:
                        result["issues"].append(f"Could not read medical text sample: {str(e)}")
                        
            except Exception as e:
                result["issues"].append(f"Error validating medical text dataset: {str(e)}")
        
        # Determine status
        if result["datasets"]:
            result["status"] = "valid" if not result["issues"] else "valid_with_issues"
        
        self.validation_results["kaggle"] = result
        logger.info(f"✅ Kaggle validation complete: {result['status']}")
    
    def validate_huggingface_data(self):
        """Validate Hugging Face datasets"""
        logger.info("📊 Validating Hugging Face datasets...")
        
        hf_dir = self.raw_dir / "huggingface"
        result = {
            "status": "missing",
            "datasets": {},
            "issues": []
        }
        
        if not hf_dir.exists():
            result["issues"].append("Hugging Face directory not found")
            self.validation_results["huggingface"] = result
            return
        
        # Check for cached datasets
        for dataset_dir in hf_dir.iterdir():
            if dataset_dir.is_dir():
                try:
                    # Count files in dataset directory
                    all_files = list(dataset_dir.rglob("*"))
                    data_files = [f for f in all_files if f.is_file() and f.suffix in ['.json', '.parquet', '.arrow']]
                    
                    result["datasets"][dataset_dir.name] = {
                        "total_files": len(all_files),
                        "data_files": len(data_files),
                        "size_mb": round(sum(f.stat().st_size for f in data_files if f.exists()) / 1024 / 1024, 2)
                    }
                except Exception as e:
                    result["issues"].append(f"Error validating {dataset_dir.name}: {str(e)}")
        
        # Determine status
        if result["datasets"]:
            result["status"] = "valid" if not result["issues"] else "valid_with_issues"
        
        self.validation_results["huggingface"] = result
        logger.info(f"✅ Hugging Face validation complete: {result['status']}")
    
    def validate_github_corpora(self):
        """Validate GitHub-sourced corpora"""
        logger.info("📊 Validating GitHub corpora...")
        
        result = {
            "status": "missing",
            "corpora": {},
            "issues": []
        }
        
        # Check medical abstracts corpus
        abstracts_dir = self.raw_dir / "medical-abstracts-tc"
        if abstracts_dir.exists():
            try:
                # Look for data files
                data_files = list(abstracts_dir.glob("**/*.txt")) + list(abstracts_dir.glob("**/*.csv"))
                result["corpora"]["medical_abstracts"] = {
                    "files": len(data_files),
                    "has_readme": (abstracts_dir / "README.md").exists(),
                    "total_size_mb": round(sum(f.stat().st_size for f in data_files) / 1024 / 1024, 2)
                }
                
                # Try to read a sample file
                if data_files:
                    sample_file = data_files[0]
                    try:
                        with open(sample_file, 'r', encoding='utf-8') as f:
                            sample_content = f.read(500)  # First 500 chars
                            result["corpora"]["medical_abstracts"]["sample_content_length"] = len(sample_content)
                    except Exception as e:
                        result["issues"].append(f"Could not read sample from {sample_file.name}: {str(e)}")
                        
            except Exception as e:
                result["issues"].append(f"Error validating medical abstracts corpus: {str(e)}")
        
        # Determine status
        if result["corpora"]:
            result["status"] = "valid" if not result["issues"] else "valid_with_issues"
        
        self.validation_results["github"] = result
        logger.info(f"✅ GitHub corpora validation complete: {result['status']}")
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        logger.info("📝 Generating validation report...")
        
        report_file = self.data_root / f"validation_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Calculate summary statistics
        total_datasets = len(self.validation_results)
        valid_datasets = sum(1 for r in self.validation_results.values() if r["status"] in ["valid", "valid_with_issues"])
        missing_datasets = sum(1 for r in self.validation_results.values() if r["status"] == "missing")
        
        summary = {
            "validation_timestamp": pd.Timestamp.now().isoformat(),
            "total_dataset_types": total_datasets,
            "valid_datasets": valid_datasets,
            "missing_datasets": missing_datasets,
            "validation_results": self.validation_results
        }
        
        # Save detailed report
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary to console
        print("\n" + "="*60)
        print("📊 DATASET VALIDATION SUMMARY")
        print("="*60)
        print(f"📈 Valid datasets: {valid_datasets}/{total_datasets}")
        print(f"❌ Missing datasets: {missing_datasets}/{total_datasets}")
        print(f"📄 Detailed report: {report_file}")
        print("\n📋 Dataset Status:")
        
        for dataset_type, result in self.validation_results.items():
            status_emoji = {"valid": "✅", "valid_with_issues": "⚠️", "missing": "❌"}
            emoji = status_emoji.get(result["status"], "❓")
            print(f"   {emoji} {dataset_type.capitalize()}: {result['status']}")
            
            # Show issues if any
            if result.get("issues"):
                for issue in result["issues"][:3]:  # Show first 3 issues
                    print(f"      └── {issue}")
        
        print("\n🎯 RECOMMENDATIONS:")
        if missing_datasets > 0:
            print("   • Run download_immediate_datasets.sh to get more data")
            print("   • Configure Kaggle API for additional datasets")
            print("   • Apply for clinical datasets (n2c2, i2b2)")
        
        if valid_datasets > 0:
            print("   • Start data preprocessing with available datasets")
            print("   • Begin model training with synthetic data")
        
        print("="*60)

def main():
    """Main validation function"""
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        data_root = "data"
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Run validation
    validator = DatasetValidator(data_root)
    results = validator.validate_all_datasets()
    
    # Exit with error code if no valid datasets found
    valid_count = sum(1 for r in results.values() if r["status"] in ["valid", "valid_with_issues"])
    if valid_count == 0:
        logger.error("❌ No valid datasets found!")
        sys.exit(1)
    else:
        logger.info(f"✅ Validation complete! {valid_count} dataset types available.")

if __name__ == "__main__":
    main()
