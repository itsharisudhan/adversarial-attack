#!/bin/bash

# Adversarial Detection System - Quick Setup Script
# 60% Milestone

echo "========================================"
echo "  Adversarial Detection System Setup"
echo "  60% Milestone Implementation"
echo "========================================"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.7+"
    exit 1
fi

echo "✓ Python found: $(python --version)"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Train models (first time only, ~2-3 hours):"
echo "   python train_all_models.py"
echo ""
echo "2. Run demo:"
echo "   python demo_60.py"
echo ""
echo "3. Start web interface:"
echo "   cd web && python app.py"
echo "   Then open: http://localhost:5000"
echo ""
echo "========================================"
