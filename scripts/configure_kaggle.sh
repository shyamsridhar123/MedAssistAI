#!/bin/bash
# configure_kaggle.sh - Interactive script to set up Kaggle API credentials

set -e

echo "🔧 Kaggle API Configuration for MedAssist AI"
echo "============================================="

# Check if Kaggle is already configured
if [ -f ~/.kaggle/kaggle.json ]; then
    echo "✅ Kaggle API already configured at ~/.kaggle/kaggle.json"
    echo "🔍 Testing connection..."
    
    if kaggle datasets list --max-size 1 &>/dev/null; then
        echo "✅ Kaggle API working correctly!"
        exit 0
    else
        echo "❌ Kaggle API not working. Let's reconfigure..."
    fi
fi

echo ""
echo "📋 To configure Kaggle API, you need to:"
echo "   1. Create a Kaggle account at https://www.kaggle.com"
echo "   2. Go to Account → Settings → API"
echo "   3. Click 'Create New API Token'"
echo "   4. Download the kaggle.json file"
echo ""

read -p "Do you have a kaggle.json file? (y/n): " has_file

if [ "$has_file" = "y" ] || [ "$has_file" = "Y" ]; then
    echo ""
    echo "📂 Please provide the path to your kaggle.json file:"
    read -p "Path: " kaggle_file_path
    
    # Expand tilde if present
    kaggle_file_path="${kaggle_file_path/#\~/$HOME}"
    
    if [ ! -f "$kaggle_file_path" ]; then
        echo "❌ File not found: $kaggle_file_path"
        exit 1
    fi
    
    # Create .kaggle directory
    mkdir -p ~/.kaggle
    
    # Copy and set permissions
    cp "$kaggle_file_path" ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
    
    echo "✅ Kaggle credentials configured!"
    
    # Test the configuration
    echo "🔍 Testing Kaggle API connection..."
    if kaggle datasets list --max-size 1 &>/dev/null; then
        echo "✅ Kaggle API working correctly!"
        echo ""
        echo "🎉 You can now download Kaggle datasets using:"
        echo "   ./scripts/download_immediate_datasets.sh"
    else
        echo "❌ Kaggle API test failed. Please check your credentials."
        exit 1
    fi
    
else
    echo ""
    echo "📝 Manual setup instructions:"
    echo "   1. Visit https://www.kaggle.com/account"
    echo "   2. Create account and verify email"
    echo "   3. Go to Account → API section"
    echo "   4. Click 'Create New API Token'"
    echo "   5. Save the downloaded kaggle.json file"
    echo "   6. Run this script again with the file path"
    echo ""
    echo "⚠️  Skipping Kaggle configuration for now."
    echo "   You can run this script later to enable Kaggle datasets."
fi
