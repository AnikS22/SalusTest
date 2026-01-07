#!/bin/bash
# SALUS HPC Sync Script
# Syncs repository to HPC using rsync

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================"
echo "SALUS HPC Sync Script"
echo "============================================================"
echo ""

# HPC configuration - EDIT THESE VALUES
HPC_HOST="${HPC_HOST:-athene-login.hpc.fau.edu}"
HPC_USER="${HPC_USER:-asahai2024}"
HPC_PATH="${HPC_PATH:-~/SalusTest}"

# Check if HPC_HOST and HPC_USER are set
if [ -z "$HPC_HOST" ] || [ -z "$HPC_USER" ]; then
    echo -e "${YELLOW}⚠️  HPC configuration not set${NC}"
    echo ""
    echo "Please set HPC_HOST and HPC_USER environment variables:"
    echo ""
    echo "  export HPC_HOST=athene-login.hpc.fau.edu"
    echo "  export HPC_USER=your_username"
    echo "  export HPC_PATH=~/SalusTest  # Optional, defaults to ~/SalusTest"
    echo ""
    echo "Or run with inline variables:"
    echo "  HPC_HOST=athene-login.hpc.fau.edu HPC_USER=your_username ./sync_to_hpc.sh"
    echo ""
    exit 1
fi

# Local directory (script location)
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Configuration:"
echo "  Local:  $LOCAL_DIR"
echo "  Remote: ${HPC_USER}@${HPC_HOST}:${HPC_PATH}"
echo ""

# Check if we can reach HPC
echo "Testing SSH connection..."
if ssh -o ConnectTimeout=5 -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" "echo 'Connection OK'" &>/dev/null; then
    echo -e "${GREEN}✓${NC} SSH connection successful"
else
    echo -e "${RED}✗${NC} Cannot connect to HPC"
    echo "  Make sure you can SSH without password (use ssh-keygen if needed)"
    echo "  Test with: ssh ${HPC_USER}@${HPC_HOST}"
    exit 1
fi
echo ""

# Rsync options
RSYNC_OPTS=(
    -avz                  # Archive mode, verbose, compress
    --progress            # Show progress
    --delete              # Delete files on remote that don't exist locally
    --exclude='.git/'     # Don't sync git directory
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='*.pth'
    --exclude='*.pt'
    --exclude='*.ckpt'
    --exclude='data/'
    --exclude='datasets/'
    --exclude='checkpoints/'
    --exclude='logs/'
    --exclude='*.log'
    --exclude='*.zarr/'
    --exclude='models/tinyvla/'
    --exclude='models/openvla/'
    --exclude='paper_data/data.zarr/'
    --exclude='paper_data/logs/'
    --exclude='test_output/'
    --exclude='tmp/'
    --exclude='temp/'
    --exclude='venv/'
    --exclude='venv_salus/'
    --exclude='.vscode/'
    --exclude='.idea/'
    --exclude='*.swp'
    --exclude='*.swo'
    --exclude='.DS_Store'
)

echo "Starting sync..."
echo ""

# Run rsync
if rsync "${RSYNC_OPTS[@]}" "$LOCAL_DIR/" "${HPC_USER}@${HPC_HOST}:${HPC_PATH}/"; then
    echo ""
    echo -e "${GREEN}✅ Sync completed successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. SSH to HPC: ssh ${HPC_USER}@${HPC_HOST}"
    echo "  2. Navigate: cd ${HPC_PATH}"
    echo "  3. Run tests: python scripts/test_hpc_phase1.py"
    echo ""
else
    echo ""
    echo -e "${RED}✗ Sync failed${NC}"
    exit 1
fi
