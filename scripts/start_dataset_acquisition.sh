#!/bin/bash
# start_dataset_acquisition.sh - Master script to begin dataset acquisition process

set -e

echo "🏥 MedAssist AI Dataset Acquisition Workflow"
echo "============================================"
echo "This script will guide you through acquiring all necessary datasets"
echo "for training the MedAssist AI clinical assistant."
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Make all scripts executable
chmod +x "$SCRIPT_DIR"/*.sh

echo "📋 Dataset Acquisition Plan:"
echo "  Phase 1: Setup environment and download immediate datasets"
echo "  Phase 2: Configure external service access (Kaggle)"
echo "  Phase 3: Generate clinical dataset applications"
echo "  Phase 4: Validate downloaded data"
echo ""

read -p "Ready to start? (y/n): " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Exiting. Run this script when ready to begin."
    exit 0
fi

echo ""
echo "🚀 Starting Phase 1: Environment Setup and Immediate Downloads"
echo "=============================================================="

# Phase 1: Setup environment
echo "📁 Setting up data environment..."
"$SCRIPT_DIR/setup_data_environment.sh"

if [ $? -ne 0 ]; then
    echo "❌ Environment setup failed. Please check the error messages above."
    exit 1
fi

echo ""
echo "⬇️  Downloading immediately available datasets..."
"$SCRIPT_DIR/download_immediate_datasets.sh"

if [ $? -ne 0 ]; then
    echo "❌ Dataset download failed. Check logs for details."
    echo "You can continue with manual downloads or troubleshooting."
fi

echo ""
echo "🔧 Starting Phase 2: External Service Configuration"
echo "=================================================="

# Phase 2: Configure Kaggle (optional)
echo "🔑 Configuring Kaggle API for additional datasets..."
"$SCRIPT_DIR/configure_kaggle.sh"

# If Kaggle was configured, try downloading Kaggle datasets again
if [ -f ~/.kaggle/kaggle.json ]; then
    echo "🔄 Re-running dataset download with Kaggle enabled..."
    "$SCRIPT_DIR/download_immediate_datasets.sh"
fi

echo ""
echo "📝 Starting Phase 3: Clinical Dataset Applications"
echo "================================================="

# Phase 3: Generate applications
echo "📋 Generating clinical dataset application materials..."
cd "$PROJECT_ROOT"
python3 "$SCRIPT_DIR/apply_for_clinical_datasets.py"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Application materials generated successfully!"
    echo "📁 Check the 'applications/' directory for:"
    echo "   • research_proposal.txt"
    echo "   • n2c2_application.txt" 
    echo "   • i2b2_application.txt"
    echo "   • application_checklist.txt"
    echo ""
    echo "🎯 Next: Customize these files with your information and submit applications"
else
    echo "⚠️  Application generation had issues, but continuing..."
fi

echo ""
echo "🔍 Starting Phase 4: Data Validation"
echo "===================================="

# Phase 4: Validate data
echo "📊 Validating downloaded datasets..."
python3 "$SCRIPT_DIR/validate_datasets.py"

echo ""
echo "🎉 Dataset Acquisition Workflow Complete!"
echo "========================================"

# Generate summary report
echo "📈 ACQUISITION SUMMARY"
echo "====================="

# Check what we got
echo "✅ Completed Steps:"
echo "   • Data environment setup"
echo "   • Immediate dataset downloads attempted"
echo "   • Kaggle configuration offered"
echo "   • Clinical dataset applications generated"
echo "   • Dataset validation completed"

echo ""
echo "📋 NEXT ACTIONS REQUIRED:"
echo "========================"

# Check if we have any data
if [ -d "$PROJECT_ROOT/data/raw/synthea" ] && [ "$(ls -A $PROJECT_ROOT/data/raw/synthea)" ]; then
    echo "✅ Synthea data available - can start development immediately"
else
    echo "⚠️  No Synthea data found - check AWS CLI configuration"
fi

if [ -f ~/.kaggle/kaggle.json ]; then
    echo "✅ Kaggle configured - additional datasets may be available"
else
    echo "📝 Configure Kaggle API for more public datasets"
fi

if [ -d "$PROJECT_ROOT/applications" ]; then
    echo "📝 Customize and submit clinical dataset applications:"
    echo "     • Edit applications/research_proposal.txt"
    echo "     • Submit n2c2 application (2-4 week approval)"
    echo "     • Submit i2b2 application (3-6 week approval)"
    echo "     • Consider MIMIC-III if needed (4-8 week approval)"
else
    echo "⚠️  Application materials not generated properly"
fi

echo ""
echo "🛠️  DEVELOPMENT RECOMMENDATIONS:"
echo "==============================="
echo "1. Start with synthetic data (Synthea) for initial development"
echo "2. Build and test Mojo-Python integration pipeline"
echo "3. Implement basic de-identification with available data"
echo "4. Submit clinical dataset applications early (long approval times)"
echo "5. Continue development while waiting for clinical data approvals"

echo ""
echo "📚 USEFUL COMMANDS:"
echo "=================="
echo "• Validate data anytime: python3 scripts/validate_datasets.py"
echo "• Re-download datasets: ./scripts/download_immediate_datasets.sh"
echo "• Check Kaggle setup: ./scripts/configure_kaggle.sh"
echo "• View logs: ls logs/"

echo ""
echo "🔗 HELPFUL LINKS:"
echo "================"
echo "• n2c2 datasets: https://n2c2.dbmi.hms.harvard.edu/data-sets"
echo "• i2b2 datasets: https://www.i2b2.org/NLP/DataSets/"
echo "• MIMIC-III: https://physionet.org/content/mimiciii/"
echo "• Synthea data: https://synthea.mitre.org/"

echo ""
echo "✨ Dataset acquisition workflow completed successfully!"
echo "   You can now begin MedAssist AI development with available data."
