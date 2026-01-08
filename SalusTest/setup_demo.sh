#!/bin/bash

# SALUS Demo Setup Script

echo "================================"
echo "SALUS Live Demo Setup"
echo "================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q rich numpy torch scipy 2>&1 | grep -v "Requirement already satisfied" || true

echo "✓ Dependencies installed"

# Check if model exists
echo ""
if [ -f "salus_fixed_pipeline.pt" ]; then
    echo "✓ SALUS model found: salus_fixed_pipeline.pt"
else
    echo "⚠ SALUS model not found: salus_fixed_pipeline.pt"
    echo "  Will run in demo mode"
fi

# Make scripts executable
chmod +x salus_live_demo.py
chmod +x salus_isaac_sim.py

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "Run the demo:"
echo "  Terminal GUI only:  python salus_live_demo.py"
echo "  With Isaac Sim:     python salus_isaac_sim.py"
echo ""
echo "Integration guide:    INTEGRATION_GUIDE.md"
echo ""
