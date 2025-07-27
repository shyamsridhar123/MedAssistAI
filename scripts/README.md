# MedAssist AI Dataset Acquisition Scripts

This directory contains automated scripts for acquiring healthcare datasets needed to train and validate the MedAssist AI clinical assistant.

## Quick Start

**Run this single command to start the complete dataset acquisition workflow:**

```bash
./scripts/start_dataset_acquisition.sh
```

This master script will guide you through all phases of dataset acquisition automatically.

## Individual Scripts

### 1. Environment Setup
- `setup_data_environment.sh` - Creates data directory structure and installs required tools
- `configure_kaggle.sh` - Interactive Kaggle API configuration

### 2. Data Download
- `download_immediate_datasets.sh` - Downloads all immediately available datasets
- `validate_datasets.py` - Validates integrity and format of downloaded data

### 3. Clinical Dataset Applications
- `apply_for_clinical_datasets.py` - Generates application materials for clinical datasets

### 4. Master Workflow
- `start_dataset_acquisition.sh` - Orchestrates the complete acquisition process

## Dataset Categories

### Immediate Access (No Registration)
- **Synthea**: 100K+ synthetic patients from AWS Open Data
- **GitHub**: Medical text corpora and abstracts
- **Hugging Face**: Open medical datasets

### Simple Registration  
- **Kaggle**: Healthcare datasets (50K records + medical text)
- **CDC/CMS**: Government public use files

### Clinical Datasets (DUA Required)
- **n2c2**: Clinical NLP challenges (2-4 week approval)
- **i2b2**: De-identification datasets (3-6 week approval)  
- **MIMIC-III**: Clinical database (4-8 week approval)

## Usage Examples

### Complete Workflow
```bash
# Run everything automatically
./scripts/start_dataset_acquisition.sh
```

### Manual Step-by-Step
```bash
# 1. Setup environment
./scripts/setup_data_environment.sh

# 2. Download immediate datasets
./scripts/download_immediate_datasets.sh

# 3. Configure Kaggle (optional)
./scripts/configure_kaggle.sh

# 4. Validate what we have
python3 scripts/validate_datasets.py

# 5. Generate clinical applications
python3 scripts/apply_for_clinical_datasets.py
```

### Validation Only
```bash
# Check current dataset status
python3 scripts/validate_datasets.py

# Re-download if needed
./scripts/download_immediate_datasets.sh
```

## Output Structure

After running the scripts, you'll have:

```
data/
├── raw/                    # Original downloaded datasets
│   ├── synthea/           # Synthetic patient records
│   ├── kaggle/            # Public healthcare datasets  
│   ├── huggingface/       # HF medical datasets
│   └── medical-abstracts-tc/  # GitHub corpora
├── processed/             # For cleaned data
├── train/val/test/        # For ML splits
└── validation_report_*.json   # Dataset validation results

applications/
├── research_proposal.txt      # Generic research proposal
├── n2c2_application.txt      # n2c2-specific application
├── i2b2_application.txt      # i2b2-specific application
└── application_checklist.txt # Step-by-step checklist

logs/
├── dataset_download_*.log    # Download logs
└── dataset_validation.log   # Validation logs
```

## Requirements

### System Requirements
- Linux/macOS/WSL (bash support)
- Python 3.7+ with pip
- Internet connection for downloads
- 10GB+ free disk space

### Python Dependencies
- pandas
- datasets (Hugging Face)
- kaggle (optional, for Kaggle datasets)
- awscli (for Synthea data)

### External Accounts (Optional)
- Kaggle account for additional datasets
- AWS CLI for larger Synthea samples
- Institutional affiliation for clinical datasets

## Troubleshooting

### Common Issues

**"AWS CLI not found"**
```bash
magic add awscli
# or
sudo apt install awscli
```

**"Kaggle API not configured"**
```bash
./scripts/configure_kaggle.sh
# Follow the interactive prompts
```

**"Python package installation errors"**
```bash
# Use magic instead of pip for this project
magic add <package_name>
# or create a virtual environment if needed
python3 -m venv venv
source venv/bin/activate
pip install <package_name>
```

**"Permission denied" on scripts**
```bash
chmod +x scripts/*.sh
```

**"No space left on device"**
- Free up at least 10GB disk space
- Consider downloading smaller dataset samples first

### Dataset-Specific Issues

**Synthea download fails:**
- Check internet connection
- Verify AWS CLI installation
- Try smaller sample size first

**Kaggle datasets unavailable:**
- Verify Kaggle account and API setup
- Check dataset availability (some may be removed)
- Try manual download from kaggle.com

**Clinical dataset applications:**
- Ensure institutional affiliation
- Have IRB approval ready
- Expect 2-8 week approval times

## Security Notes

⚠️ **Important**: Clinical datasets contain sensitive information

- All clinical data downloads are encrypted and logged
- Follow your institution's data handling policies  
- Use secure computing environments only
- Delete data per DUA requirements when project ends
- Never commit clinical data to version control

## Support

For issues with these scripts:
1. Check the logs in `logs/` directory
2. Run validation: `python3 scripts/validate_datasets.py`
3. Review the troubleshooting section above
4. Check individual script documentation

For dataset access issues:
- n2c2: Contact through their website
- i2b2: Contact through their website  
- MIMIC: physionet-support@mit.edu
- Kaggle: kaggle.com support
