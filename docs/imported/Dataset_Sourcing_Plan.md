# Dataset Sourcing Plan for MedAssist AI

## Overview
This plan outlines the systematic acquisition of healthcare datasets needed for training and validating the three core components of MedAssist AI: PHI de-identification, clinical summarization, and diagnostic assistance.

## Priority 1: Immediate Access Datasets (Week 1)

### Synthetic Healthcare Data (No Registration Required)

#### 1. Synthea™ Synthetic Patient Data
- **Access**: Direct download from AWS Open Data Registry
- **Command**: `aws s3 sync --no-sign-request s3://synthea-omop/1k_person_sample/ ./data/synthea/`
- **Content**: 1K, 100K, and 2.8M synthetic patients in OMOP format
- **Use Cases**: 
  - Initial development and testing without privacy concerns
  - Clinical summarization training data
  - Diagnostic assistant baseline training
- **Size**: ~2GB for 100K patients
- **Format**: CSV, JSON, FHIR

#### 2. Kaggle Healthcare Datasets  
- **Access**: Free Kaggle account signup
- **Key Datasets**:
  - Healthcare Dataset (50K synthetic records): `kaggle datasets download prasad22/healthcare-dataset`
  - Medical Text Classification (14,438 abstracts): `kaggle datasets download chaitanyakck/medical-text`
- **Use Cases**:
  - Text classification benchmarks
  - Clinical terminology validation
  - Diagnostic category training

### Government Open Data

#### 3. CDC Public Use Files
- **Access**: Direct download, no DUA required
- **URL**: https://data.cdc.gov/
- **Key Datasets**:
  - NHANES (National Health and Nutrition Examination Survey)
  - NHIS (National Health Interview Survey)
  - BRFSS (Behavioral Risk Factor Surveillance System)
- **Use Cases**: Population health baselines, demographic data

## Priority 2: Clinical Text Datasets (Week 2-3)

### 4. n2c2 Clinical NLP Datasets
- **Access**: Data Use Agreement (DUA) required, no course prerequisites
- **URL**: https://n2c2.dbmi.hms.harvard.edu/data-sets
- **Critical Datasets**:
  - **2014 De-identification Challenge**: Ground truth PHI annotations
  - **2018 Cohort Selection**: Clinical reasoning tasks
  - **2019 Medication Extraction**: Structured clinical information
- **Use Cases**: 
  - Primary training data for PHI detection (target: 0.9732 F1-score)
  - Clinical NER and concept extraction
- **Timeline**: 2-4 weeks for DUA approval

#### DUA Application Process:
1. Register at n2c2 portal
2. Submit research proposal (attach our PRD)
3. Sign institutional DUA
4. Wait for approval (typically 2-4 weeks)

### 5. i2b2 De-identification Dataset
- **Access**: Research application required
- **URL**: https://www.i2b2.org/NLP/DataSets/
- **Content**: Annotated clinical notes with PHI labels
- **Use Cases**: Gold standard for de-identification training and evaluation
- **Timeline**: 3-6 weeks for approval

## Priority 3: Advanced Clinical Data (Week 4-6)

### 6. Stanford AIMI Shared Datasets
- **Access**: Click-through license agreement
- **URL**: https://aimi.stanford.edu/shared-datasets
- **Content**: 1M+ medical images with annotations
- **Use Cases**: Future multimodal capabilities, imaging-text integration
- **Size**: Several TB of data

### 7. MIMIC-III Clinical Database (If Needed)
- **Access**: CITI training + PhysioNet credentialing
- **Requirements**: 
  - Complete CITI "Data or Specimens Only Research" course
  - Institutional affiliation verification
  - Detailed research proposal
- **Content**: De-identified clinical notes, vital signs, medications
- **Use Cases**: Advanced clinical summarization, longitudinal patient tracking
- **Timeline**: 4-8 weeks for credentialing

## Implementation Timeline

### Week 1: Foundation Setup
```bash
# Setup data directory structure
mkdir -p data/{raw,processed,train,val,test}
mkdir -p data/raw/{synthea,kaggle,cdc,n2c2,i2b2,stanford}

# Download immediate access datasets
aws s3 sync --no-sign-request s3://synthea-omop/1k_person_sample/ data/raw/synthea/
kaggle datasets download prasad22/healthcare-dataset -p data/raw/kaggle/
kaggle datasets download chaitanyakck/medical-text -p data/raw/kaggle/
```

### Week 2-3: DUA Submissions
- Submit n2c2 DUA application
- Submit i2b2 research application  
- Begin Stanford AIMI registration
- Start preprocessing synthetic data

### Week 4-6: Advanced Dataset Integration
- Process approved clinical datasets
- Implement data validation pipelines
- Create train/val/test splits
- Begin model training with available data

## Data Organization Structure

```
data/
├── raw/                          # Original downloaded datasets
│   ├── synthea/                 # Synthetic patient records
│   ├── kaggle/                  # Public healthcare datasets
│   ├── cdc/                     # Government public use files
│   ├── n2c2/                    # Clinical NLP challenge data
│   ├── i2b2/                    # De-identification datasets
│   └── stanford/                # AIMI medical imaging
├── processed/                   # Cleaned and normalized data
│   ├── deid_training/          # PHI detection training sets
│   ├── summarization/          # Clinical summarization data
│   └── diagnostic/             # Diagnostic assistance training
├── train/                      # Training splits (70%)
├── val/                        # Validation splits (15%)
└── test/                       # Test splits (15%)
```

## Dataset Mapping to Model Components

### De-identification Engine Training
- **Primary**: n2c2 2014 De-identification Challenge
- **Secondary**: i2b2 de-identification datasets
- **Validation**: Synthea synthetic notes (known clean baseline)
- **Target**: ≥99% recall, ≥95% precision for 18 HIPAA identifiers

### Clinical Summarization Model
- **Primary**: Synthea longitudinal patient records
- **Secondary**: Medical abstracts from Kaggle
- **Enhancement**: n2c2 clinical reasoning datasets
- **Target**: Clinically accurate summaries with preserved terminology

### Diagnostic Assistant Model  
- **Primary**: Synthea diagnosis codes and patient histories
- **Secondary**: n2c2 cohort selection and reasoning tasks
- **Validation**: Medical text classification datasets
- **Target**: Evidence-based suggestions with confidence scoring

## Legal and Compliance Considerations

### Data Use Agreements (DUAs)
- **Track all DUA requirements** and expiration dates
- **Document data provenance** for audit purposes
- **Ensure HIPAA compliance** for any real clinical data
- **Implement data retention policies** per agreement terms

### Security Requirements
- **Encrypt all datasets** at rest using AES-256
- **Access control** limited to project team members
- **Audit logging** for all data access and processing
- **Secure deletion** procedures for temporary processing files

## Success Metrics

### Data Acquisition Goals
- **Week 1**: 100K+ synthetic patient records available
- **Week 3**: 50K+ clinical text samples with PHI annotations
- **Week 6**: Complete training datasets for all three model components
- **Week 8**: Validated data pipelines producing clean train/val/test splits

### Quality Benchmarks
- **Data Coverage**: All 18 HIPAA identifiers represented in training data
- **Clinical Diversity**: Multiple medical specialties and note types
- **Volume Targets**: 
  - De-identification: 10K+ annotated clinical notes
  - Summarization: 50K+ patient records with outcomes
  - Diagnostics: 25K+ diagnosis-symptom pairs

## Risk Mitigation

### Backup Data Sources
- **If n2c2/i2b2 delayed**: Focus on synthetic data + medical abstracts
- **If MIMIC unavailable**: Use Stanford AIMI + public datasets
- **If DUAs rejected**: Proceed with synthetic-only initial version

### Technical Contingencies
- **Data quality issues**: Implement robust validation pipelines
- **Format inconsistencies**: Build flexible data preprocessing
- **Privacy concerns**: Default to synthetic data for development

This plan ensures we can begin development immediately with synthetic data while systematically adding real clinical datasets to improve model performance and clinical relevance.
