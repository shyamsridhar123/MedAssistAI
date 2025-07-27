#!/bin/bash
# download_immediate_datasets.sh - Download datasets available without registration

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
LOG_FILE="$PROJECT_ROOT/logs/dataset_download_$(date +%Y%m%d_%H%M%S).log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "🚀 Starting immediate dataset downloads for MedAssist AI"

# Ensure data directories exist
if [ ! -d "$DATA_DIR" ]; then
    log "❌ Data directory not found. Run setup_data_environment.sh first"
    exit 1
fi

# Function to check available disk space (in GB)
check_disk_space() {
    local required_gb=$1
    local available_gb=$(df "$DATA_DIR" | awk 'NR==2 {print int($4/1024/1024)}')
    
    if [ "$available_gb" -lt "$required_gb" ]; then
        log "❌ Insufficient disk space. Required: ${required_gb}GB, Available: ${available_gb}GB"
        exit 1
    fi
    
    log "✅ Disk space check passed. Available: ${available_gb}GB"
}

# Check for at least 10GB free space
check_disk_space 10

# 1. Download Synthea Synthetic Patient Data
log "📥 Downloading Synthea synthetic patient data..."
cd "$DATA_DIR/raw/synthea"

if [ ! -f "download_complete.flag" ]; then
    log "⬇️  Downloading 1K patient sample..."
    if magic run aws s3 sync --no-sign-request s3://synthea-omop/1k_person_sample/ ./1k_sample/ --exclude "*" --include "*.csv" 2>&1 | tee -a "$LOG_FILE"; then
        log "✅ 1K sample downloaded successfully"
        
        # Try to download 100K sample if space permits
        log "⬇️  Attempting 100K patient sample download..."
        if magic run aws s3 sync --no-sign-request s3://synthea-omop/100k_person_sample/ ./100k_sample/ --exclude "*" --include "*.csv" 2>&1 | tee -a "$LOG_FILE"; then
            log "✅ 100K sample downloaded successfully"
        else
            log "⚠️  100K sample download failed or incomplete - continuing with 1K sample"
        fi
        
        touch download_complete.flag
    else
        log "❌ Synthea download failed"
        exit 1
    fi
else
    log "✅ Synthea data already downloaded"
fi

# 2. Download Kaggle Datasets 
log "📥 Downloading Kaggle datasets..."
cd "$DATA_DIR/raw/kaggle"

if [ -f ~/.kaggle/kaggle.json ]; then
    log "✅ Kaggle API configured, downloading datasets..."
    
    # Healthcare Dataset (50K synthetic records)
    if [ ! -d "healthcare-dataset" ]; then
        log "⬇️  Downloading healthcare dataset..."
        if magic run kaggle datasets download prasad22/healthcare-dataset --unzip 2>&1 | tee -a "$LOG_FILE"; then
            log "✅ Healthcare dataset downloaded"
        else
            log "⚠️  Healthcare dataset download failed"
        fi
    else
        log "✅ Healthcare dataset already exists"
    fi
    
    # Medical Text Classification
    if [ ! -d "medical-text" ]; then
        log "⬇️  Downloading medical text classification dataset..."
        if magic run kaggle datasets download chaitanyakck/medical-text --unzip 2>&1 | tee -a "$LOG_FILE"; then
            log "✅ Medical text dataset downloaded"
        else
            log "⚠️  Medical text dataset download failed"
        fi
    else
        log "✅ Medical text dataset already exists"
    fi
    
else
    log "⚠️  Kaggle API not configured. Skipping Kaggle datasets."
    log "ℹ️  To enable Kaggle downloads:"
    log "   1. Sign up at kaggle.com"
    log "   2. Go to Account → API → Create New API Token"
    log "   3. Place kaggle.json in ~/.kaggle/"
    log "   4. chmod 600 ~/.kaggle/kaggle.json"
fi

# 3. Download Hugging Face Healthcare Datasets
log "📥 Downloading Hugging Face medical datasets..."
cd "$DATA_DIR/raw/huggingface"

# Create a temporary Python script for Hugging Face downloads
cat > /tmp/download_hf_datasets.py << 'PYTHON_EOF'
import os
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)

try:
    # Medical text classification dataset
    print("Downloading medical abstracts dataset...")
    dataset = load_dataset("medical_questions_pairs", cache_dir="./medical_questions")
    print(f"✅ Medical questions dataset: {len(dataset['train'])} samples")
    
    # Clinical NER dataset (if available)
    print("Downloading clinical NER dataset...")
    try:
        ner_dataset = load_dataset("clinical-ner", cache_dir="./clinical_ner")
        print(f"✅ Clinical NER dataset downloaded")
    except:
        print("⚠️  Clinical NER dataset not available")
    
    # Medical conversation dataset
    print("Downloading medical conversation dataset...")
    try:
        conv_dataset = load_dataset("medical_dialogue", cache_dir="./medical_dialogue")
        print(f"✅ Medical dialogue dataset downloaded")
    except:
        print("⚠️  Medical dialogue dataset not available")
        
except Exception as e:
    print(f"⚠️  Some Hugging Face datasets unavailable: {e}")
    
print("✅ Hugging Face download attempts completed")
PYTHON_EOF

# Run the Python script using magic
log "⬇️  Running Hugging Face dataset download script..."
magic run python3 /tmp/download_hf_datasets.py 2>&1 | tee -a "$LOG_FILE"
rm -f /tmp/download_hf_datasets.py

# 4. Download Public Medical Text Corpora
log "📥 Downloading additional public medical corpora..."
cd "$DATA_DIR/raw"

# Medical text corpus from GitHub
if [ ! -d "medical-abstracts-tc" ]; then
    log "⬇️  Downloading medical abstracts classification corpus..."
    git clone https://github.com/sebischair/Medical-Abstracts-TC-Corpus.git medical-abstracts-tc 2>&1 | tee -a "$LOG_FILE"
    if [ $? -eq 0 ]; then
        log "✅ Medical abstracts corpus cloned successfully"
    else
        log "⚠️  Failed to clone medical abstracts corpus"
    fi
else
    log "✅ Medical abstracts corpus already exists"
fi

# 5. Generate download summary
log "📊 Generating download summary..."

SUMMARY_FILE="$DATA_DIR/download_summary_$(date +%Y%m%d).txt"
cat > "$SUMMARY_FILE" << EOF
MedAssist AI Dataset Download Summary
Generated: $(date)

DOWNLOADED DATASETS:
==================

Synthea Synthetic Data:
- Location: data/raw/synthea/
- 1K Sample: $(if [ -d "$DATA_DIR/raw/synthea/1k_sample" ]; then echo "✅ Available"; else echo "❌ Missing"; fi)
- 100K Sample: $(if [ -d "$DATA_DIR/raw/synthea/100k_sample" ]; then echo "✅ Available"; else echo "❌ Missing"; fi)

Kaggle Datasets:
- Healthcare Dataset: $(if [ -d "$DATA_DIR/raw/kaggle/healthcare-dataset" ]; then echo "✅ Available"; else echo "❌ Missing"; fi)
- Medical Text: $(if [ -d "$DATA_DIR/raw/kaggle/medical-text" ]; then echo "✅ Available"; else echo "❌ Missing"; fi)

Hugging Face Datasets:
- Location: data/raw/huggingface/
- Medical Questions: $(if [ -d "$DATA_DIR/raw/huggingface/medical_questions" ]; then echo "✅ Available"; else echo "❌ Missing"; fi)

GitHub Corpora:
- Medical Abstracts TC: $(if [ -d "$DATA_DIR/raw/medical-abstracts-tc" ]; then echo "✅ Available"; else echo "❌ Missing"; fi)

NEXT STEPS:
===========
1. Run data validation: python scripts/validate_datasets.py
2. Apply for clinical datasets: n2c2, i2b2 (see docs/Dataset_Sourcing_Plan.md)
3. Start data preprocessing: python scripts/preprocess_synthetic_data.py

Total disk usage: $(du -sh "$DATA_DIR" 2>/dev/null | cut -f1 || echo "Unknown")
EOF

log "📝 Download summary saved to: $SUMMARY_FILE"
cat "$SUMMARY_FILE"

log "🎉 Immediate dataset download complete!"
log "📋 Full log available at: $LOG_FILE"

# Return to original directory
cd "$PROJECT_ROOT"
