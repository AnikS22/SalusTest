#!/bin/bash
# Sync data FROM HPC back to local machine
# Run this ON YOUR LOCAL MACHINE

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "============================================================"
echo "Sync Data FROM HPC to Local"
echo "============================================================"
echo ""

# HPC configuration - EDIT THESE VALUES
HPC_HOST="${HPC_HOST:-}"
HPC_USER="${HPC_USER:-}"
HPC_DATA_DIR="${HPC_DATA_DIR:-~/salus_data_temporal}"
HPC_CHECKPOINT_DIR="${HPC_CHECKPOINT_DIR:-~/SalusTest/checkpoints}"

LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-$HOME/salus_data_hpc}"
LOCAL_CHECKPOINT_DIR="${LOCAL_CHECKPOINT_DIR:-$HOME/SalusTest/checkpoints_hpc}"

# Check configuration
if [ -z "$HPC_HOST" ] || [ -z "$HPC_USER" ]; then
    echo -e "${YELLOW}⚠️  HPC configuration not set${NC}"
    echo ""
    echo "Set environment variables:"
    echo "  export HPC_HOST=athene.informatik.tu-darmstadt.de"
    echo "  export HPC_USER=your_username"
    echo ""
    echo "Or run with inline variables:"
    echo "  HPC_HOST=athene.informatik.tu-darmstadt.de HPC_USER=your_username ./sync_from_hpc.sh"
    echo ""
    exit 1
fi

echo "Configuration:"
echo "  HPC: ${HPC_USER}@${HPC_HOST}"
echo "  HPC data: ${HPC_DATA_DIR}"
echo "  HPC checkpoints: ${HPC_CHECKPOINT_DIR}"
echo "  Local data: ${LOCAL_DATA_DIR}"
echo "  Local checkpoints: ${LOCAL_CHECKPOINT_DIR}"
echo ""

# Check SSH connection
echo "Testing SSH connection..."
if ssh -o ConnectTimeout=5 -o BatchMode=yes "${HPC_USER}@${HPC_HOST}" "echo 'OK'" &>/dev/null; then
    echo -e "${GREEN}✓${NC} SSH connection successful"
else
    echo -e "${RED}✗${NC} Cannot connect to HPC"
    exit 1
fi
echo ""

# Function to sync directory
sync_dir() {
    local remote_path="$1"
    local local_path="$2"
    local name="$3"

    echo "============================================================"
    echo "Syncing $name..."
    echo "============================================================"
    echo ""

    # Check if remote directory exists
    if ! ssh "${HPC_USER}@${HPC_HOST}" "[ -d $remote_path ]"; then
        echo -e "${YELLOW}⚠${NC} $name not found on HPC at: $remote_path"
        echo "Skipping..."
        echo ""
        return
    fi

    # Get remote directory size
    echo "Checking remote size..."
    remote_size=$(ssh "${HPC_USER}@${HPC_HOST}" "du -sh $remote_path" | cut -f1)
    echo "  Remote size: $remote_size"
    echo ""

    # Create local directory
    mkdir -p "$local_path"

    # Ask for confirmation
    read -p "Sync $name ($remote_size)? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping $name"
        echo ""
        return
    fi

    echo "Syncing..."
    rsync -avz --progress \
        --exclude="*.pyc" \
        --exclude="__pycache__/" \
        "${HPC_USER}@${HPC_HOST}:${remote_path}/" \
        "${local_path}/"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $name synced successfully"
        local_size=$(du -sh "$local_path" | cut -f1)
        echo "  Local size: $local_size"
    else
        echo -e "${RED}✗${NC} Failed to sync $name"
    fi
    echo ""
}

# Sync data
sync_dir "$HPC_DATA_DIR" "$LOCAL_DATA_DIR" "Training data"

# Sync checkpoints
sync_dir "$HPC_CHECKPOINT_DIR" "$LOCAL_CHECKPOINT_DIR" "Model checkpoints"

# Summary
echo "============================================================"
echo "Sync Complete"
echo "============================================================"
echo ""

if [ -d "$LOCAL_DATA_DIR" ]; then
    echo "Local data directory:"
    du -sh "$LOCAL_DATA_DIR"
    ls -lh "$LOCAL_DATA_DIR"
fi
echo ""

if [ -d "$LOCAL_CHECKPOINT_DIR" ]; then
    echo "Local checkpoints:"
    du -sh "$LOCAL_CHECKPOINT_DIR"
    ls -lh "$LOCAL_CHECKPOINT_DIR"
fi
echo ""

echo -e "${GREEN}✅ All data synced to local machine${NC}"
echo ""
