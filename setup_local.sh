#!/bin/bash
# SALUS Local Machine Setup Script
# Run this on your 4x RTX 2080 Ti machine

set -e  # Exit on error

echo "=========================================="
echo "SALUS/GUARDIAN Local Machine Setup"
echo "4x RTX 2080 Ti Configuration"
echo "=========================================="
echo ""

# Get current directory
SALUS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SALUS_DIR"

echo "Working directory: $SALUS_DIR"
echo ""

# Step 1: Check NVIDIA GPU
echo "[1/8] Checking NVIDIA GPUs..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found! Install NVIDIA drivers first."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
echo "✅ Found $GPU_COUNT GPUs"

if [ "$GPU_COUNT" != "4" ]; then
    echo "⚠️  Warning: Expected 4 GPUs, found $GPU_COUNT"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Step 2: Check Python
echo "[2/8] Checking Python..."
if ! command -v python3.10 &> /dev/null; then
    echo "⚠️  Python 3.10 not found, trying python3..."
    if command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
    else
        echo "❌ Python 3 not found! Install Python 3.10+"
        exit 1
    fi
else
    PYTHON_CMD=python3.10
fi

PYTHON_VERSION=$($PYTHON_CMD --version)
echo "✅ Using: $PYTHON_VERSION"
echo ""

# Step 3: Create virtual environment
echo "[3/8] Creating virtual environment..."
if [ ! -d "venv_salus" ]; then
    $PYTHON_CMD -m venv venv_salus
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

source venv_salus/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Step 4: Install PyTorch
echo "[4/8] Installing PyTorch with CUDA 11.8..."
pip install --upgrade pip setuptools wheel -q

echo "  Installing PyTorch (this may take 5-10 minutes)..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118 -q

echo "✅ PyTorch installed"
echo ""

# Step 5: Test GPU access
echo "[5/8] Testing PyTorch GPU access..."
python << 'PYEOF'
import torch
print("="*60)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
print("-"*60)

if not torch.cuda.is_available():
    print("❌ CUDA not available!")
    exit(1)

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name}")
    print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")

print("="*60)

# Test allocation
try:
    for i in range(min(4, torch.cuda.device_count())):
        t = torch.randn(100, 100).cuda(i)
    print("✅ All GPUs accessible!")
except Exception as e:
    print(f"❌ GPU allocation failed: {e}")
    exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo "❌ PyTorch GPU test failed!"
    exit 1
fi
echo ""

# Step 6: Install dependencies
echo "[6/8] Installing dependencies..."
pip install numpy scipy scikit-learn pandas \
    opencv-python pillow h5py tqdm wandb tensorboard \
    matplotlib seaborn -q

echo "✅ Dependencies installed"
echo ""

# Step 7: Create directory structure
echo "[7/8] Creating project structure..."

# Core directories
mkdir -p salus/{core/{predictor,manifold,synthesis,vla},data,training,deployment,simulation,utils,scripts,tests,models/{predictor,manifold,dynamics},notebooks,config}

# Data directories
mkdir -p data/{raw_episodes,processed,labels,checkpoints}

# Logs
mkdir -p logs/{training,simulation,deployment}

# Create __init__.py files
cat > salus/__init__.py << 'EOF'
"""
SALUS/GUARDIAN: Predictive Runtime Safety for VLA Models
Copyright (c) 2025 - Proprietary
"""
__version__ = "0.1.0"
EOF

for dir in core core/predictor core/manifold core/synthesis core/vla data training deployment simulation utils; do
    echo "\"\"\"$(basename $dir) module\"\"\"" > salus/$dir/__init__.py
done

echo "✅ Project structure created"
echo ""

# Step 8: Save requirements
echo "[8/8] Saving requirements..."
pip freeze > requirements.txt
echo "✅ Requirements saved to requirements.txt"
echo ""

# Summary
echo "=========================================="
echo "✅ SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Download TinyVLA model:"
echo "   cd ~/"
echo "   git clone https://github.com/OpenDriveLab/TinyVLA.git"
echo "   cd TinyVLA && pip install -e ."
echo "   huggingface-cli download TinyVLA/tinyvla-1b --local-dir ~/models/tinyvla/tinyvla-1b"
echo ""
echo "2. Implement VLA wrapper:"
echo "   See: $SALUS_DIR/LOCAL_MACHINE_SETUP.md (Section: PHASE 4)"
echo ""
echo "3. Start data collection:"
echo "   python scripts/collect_data_local.py"
echo ""
echo "For detailed instructions, see:"
echo "  - LOCAL_MACHINE_SETUP.md"
echo "  - GETTING_STARTED.md"
echo ""
echo "=========================================="
