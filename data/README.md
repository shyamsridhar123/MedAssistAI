# MedAssist AI Dataset Storage

This directory contains healthcare datasets for training and validation.

## Structure
- `raw/`: Original downloaded datasets
- `processed/`: Cleaned and normalized data  
- `train/`: Training splits (70%)
- `val/`: Validation splits (15%)
- `test/`: Test splits (15%)

## Security Notes
- All clinical data must be encrypted at rest
- Follow HIPAA compliance requirements
- Audit all data access in logs/
- Use secure deletion for temporary files

## Dataset Sources
See ../docs/Dataset_Sourcing_Plan.md for complete sourcing information.
