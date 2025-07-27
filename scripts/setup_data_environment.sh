#!/bin/bash
# setup_data_environment.sh - Initialize data directory structure and environment

set -e  # Exit on any error

echo "🏗️  Setting up MedAssist AI data environment..."

# Create data directory structure
echo "📁 Creating data directory structure..."
mkdir -p data/{raw,processed,train,val,test}
mkdir -p data/raw/{synthea,kaggle,cdc,n2c2,i2b2,stanford,huggingface}
mkdir -p data/processed/{deid_training,summarization,diagnostic}
mkdir -p logs

echo "✅ Data directories created successfully!"

# Check for required tools
echo "🔧 Checking for required tools..."

# Check for AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Installing with magic..."
    magic add awscli
    if [ $? -ne 0 ]; then
        echo "⚠️  Magic install failed. You can install manually later:"
        echo "   magic add awscli"
        echo "   or: sudo apt install awscli"
    fi
else
    echo "✅ AWS CLI found"
fi

# Check for Kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo "❌ Kaggle CLI not found. Installing with magic..."
    magic add kaggle
    if [ $? -ne 0 ]; then
        echo "⚠️  Magic install failed. You can install manually later:"
        echo "   magic add kaggle"
        echo "   or: magic add python-kaggle"
    fi
else
    echo "✅ Kaggle CLI found"
fi

# Check for basic tools
for tool in curl wget unzip; do
    if ! command -v $tool &> /dev/null; then
        echo "❌ $tool not found. Please install it manually."
        exit 1
    else
        echo "✅ $tool found"
    fi
done

# Create .gitignore for data directory if it doesn't exist
if [ ! -f data/.gitignore ]; then
    echo "📝 Creating data/.gitignore..."
    cat > data/.gitignore << EOF
# Ignore all data files for privacy and size
*
!.gitignore
!README.md
EOF
fi

# Create data README
cat > data/README.md << EOF
# MedAssist AI Dataset Storage

This directory contains healthcare datasets for training and validation.

## Structure
- \`raw/\`: Original downloaded datasets
- \`processed/\`: Cleaned and normalized data  
- \`train/\`: Training splits (70%)
- \`val/\`: Validation splits (15%)
- \`test/\`: Test splits (15%)

## Security Notes
- All clinical data must be encrypted at rest
- Follow HIPAA compliance requirements
- Audit all data access in logs/
- Use secure deletion for temporary files

## Dataset Sources
See ../docs/Dataset_Sourcing_Plan.md for complete sourcing information.
EOF

echo "🎉 Data environment setup complete!"
echo "📖 Next steps:"
echo "   1. Run ./scripts/download_immediate_datasets.sh for synthetic data"
echo "   2. Configure Kaggle API credentials if needed"
echo "   3. Apply for clinical dataset access (n2c2, i2b2)"
