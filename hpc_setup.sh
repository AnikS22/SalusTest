#!/bin/bash
# SALUS HPC Setup Script
# Automated setup for A100 deployment

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SALUS HPC A100 Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# Step 1: Verify GPU
echo -e "\n${YELLOW}Step 1: Verifying GPU...${NC}"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo -e "  Detected GPU: ${BLUE}$GPU_NAME${NC}"

if [[ $GPU_NAME =~ "A100" ]]; then
    echo -e "  ${GREEN}✓ A100 detected!${NC}"
else
    echo -e "  ${RED}⚠ WARNING: Expected A100, got $GPU_NAME${NC}"
    read -p "  Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 2: Check conda
echo -e "\n${YELLOW}Step 2: Checking conda installation...${NC}"
if command -v conda &> /dev/null; then
    CONDA_VERSION=$(conda --version)
    echo -e "  ${GREEN}✓ Conda found: $CONDA_VERSION${NC}"
else
    echo -e "  ${RED}✗ Conda not found${NC}"
    echo -e "  Installing Miniconda..."

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p ~/miniconda3
    rm ~/miniconda.sh

    # Initialize conda
    ~/miniconda3/bin/conda init bash
    source ~/.bashrc

    echo -e "  ${GREEN}✓ Conda installed${NC}"
    echo -e "  ${YELLOW}⚠ Please run: source ~/.bashrc${NC}"
    echo -e "  ${YELLOW}   Then re-run this script${NC}"
    exit 0
fi

# Step 3: Create/activate isaaclab environment
echo -e "\n${YELLOW}Step 3: Setting up Isaac Lab environment...${NC}"

if conda env list | grep -q "isaaclab"; then
    echo -e "  ${GREEN}✓ isaaclab environment exists${NC}"
else
    echo -e "  Creating isaaclab environment..."
    conda create -n isaaclab python=3.10 -y
    echo -e "  ${GREEN}✓ Environment created${NC}"
fi

# Activate environment
echo -e "  Activating isaaclab environment..."
source ~/miniconda3/etc/profile.d/conda.sh || source ~/miniconda/etc/profile.d/conda.sh
conda activate isaaclab

echo -e "  ${GREEN}✓ Environment activated${NC}"

# Step 4: Install Isaac Lab (if not already installed)
echo -e "\n${YELLOW}Step 4: Checking Isaac Lab installation...${NC}"

if python -c "import isaaclab" 2>/dev/null; then
    echo -e "  ${GREEN}✓ Isaac Lab already installed${NC}"
else
    echo -e "  ${RED}✗ Isaac Lab not found${NC}"
    echo -e "  ${YELLOW}Please install Isaac Lab first:${NC}"
    echo -e "    1. Clone: git clone https://github.com/isaac-sim/IsaacLab.git"
    echo -e "    2. Follow install instructions in IsaacLab/docs/source/setup/installation.rst"
    echo -e "    3. Re-run this setup script"
    exit 1
fi

# Step 5: Install SALUS dependencies
echo -e "\n${YELLOW}Step 5: Installing SALUS dependencies...${NC}"

pip install -q zarr torch torchvision tqdm pyyaml numpy

# Check for lerobot (SmolVLA)
if python -c "import lerobot" 2>/dev/null; then
    echo -e "  ${GREEN}✓ lerobot (SmolVLA) already installed${NC}"
else
    echo -e "  Installing lerobot for SmolVLA..."
    pip install -q lerobot
    echo -e "  ${GREEN}✓ lerobot installed${NC}"
fi

echo -e "  ${GREEN}✓ All dependencies installed${NC}"

# Step 6: Create directory structure
echo -e "\n${YELLOW}Step 6: Creating directory structure...${NC}"

mkdir -p ~/salus_a100/{a100_data,a100_checkpoints,a100_logs,a100_results,backup_logs}
cd ~/salus_a100

echo -e "  ${GREEN}✓ Directories created${NC}"
echo -e "  Working directory: ${BLUE}$(pwd)${NC}"

# Step 7: Verify SALUS codebase
echo -e "\n${YELLOW}Step 7: Verifying SALUS codebase...${NC}"

if [ -d "salus" ]; then
    echo -e "  ${GREEN}✓ SALUS codebase found${NC}"
else
    echo -e "  ${RED}✗ SALUS codebase not found in $(pwd)${NC}"
    echo -e "  ${YELLOW}Please copy SALUS codebase to: ~/salus_a100/${NC}"
    echo -e "  ${YELLOW}From your local machine, run:${NC}"
    echo -e "    ${BLUE}scp -r /path/to/SalusTest username@hpc_address:~/salus_a100/${NC}"
    exit 1
fi

# Step 8: Download SmolVLA model (if needed)
echo -e "\n${YELLOW}Step 8: Checking SmolVLA model...${NC}"

MODEL_DIR=~/models/smolvla/smolvla_base
if [ -d "$MODEL_DIR" ]; then
    echo -e "  ${GREEN}✓ SmolVLA model found${NC}"
else
    echo -e "  ${YELLOW}Downloading SmolVLA model...${NC}"
    echo -e "  ${YELLOW}(This may take 5-10 minutes)${NC}"
    mkdir -p ~/models/smolvla
    cd ~/models/smolvla

    # Download using huggingface-cli
    pip install -q huggingface_hub
    huggingface-cli download HuggingFaceTB/smolvla-base --local-dir smolvla_base

    echo -e "  ${GREEN}✓ SmolVLA model downloaded${NC}"
    cd ~/salus_a100
fi

# Step 9: Summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ HPC Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}System Information:${NC}"
echo -e "  GPU: $GPU_NAME"
echo -e "  Python: $(python --version 2>&1)"
echo -e "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo -e "  CUDA: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"
echo -e "  Isaac Lab: $(python -c 'import isaaclab; print(isaaclab.__version__)' 2>/dev/null || echo 'Installed')"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "  1. Install Rclone for Google Cloud backup"
echo -e "     ${BLUE}bash setup_rclone.sh${NC}"
echo -e ""
echo -e "  2. Test SALUS with 1 episode"
echo -e "     ${BLUE}./test_salus.sh${NC}"
echo -e ""
echo -e "  3. Start full data collection (500 episodes, ~50 hours)"
echo -e "     ${BLUE}./deploy_a100.sh collect 500 8${NC}"

echo -e "\n${GREEN}Ready for deployment!${NC}"
