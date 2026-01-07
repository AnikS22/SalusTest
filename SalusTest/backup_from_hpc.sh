#!/bin/bash
# Backup SALUS data from HPC to cloud storage
# Run this ON THE HPC CLUSTER

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "============================================================"
echo "SALUS HPC Data Backup"
echo "============================================================"
echo ""

# Configuration
RCLONE_REMOTE="${RCLONE_REMOTE:-backup}"  # Name of rclone remote
DATA_DIR="${DATA_DIR:-$HOME/salus_data_temporal}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/SalusTest/checkpoints}"
BACKUP_PATH="${BACKUP_PATH:-SALUS_HPC_Backup}"

# Check if rclone is installed
if ! command -v rclone &> /dev/null && ! [ -f "$HOME/bin/rclone" ]; then
    echo -e "${RED}✗ rclone not found${NC}"
    echo ""
    echo "Install rclone first:"
    echo "  bash setup_rclone_hpc.sh"
    echo ""
    exit 1
fi

# Use rclone from ~/bin if available
if [ -f "$HOME/bin/rclone" ]; then
    RCLONE="$HOME/bin/rclone"
else
    RCLONE="rclone"
fi

echo "Configuration:"
echo "  Remote: ${RCLONE_REMOTE}"
echo "  Data dir: ${DATA_DIR}"
echo "  Checkpoint dir: ${CHECKPOINT_DIR}"
echo "  Backup path: ${RCLONE_REMOTE}:${BACKUP_PATH}"
echo ""

# Test rclone connection
echo "Testing rclone connection..."
if $RCLONE lsd "${RCLONE_REMOTE}:" &>/dev/null; then
    echo -e "${GREEN}✓${NC} Connected to ${RCLONE_REMOTE}"
else
    echo -e "${RED}✗${NC} Cannot connect to ${RCLONE_REMOTE}"
    echo ""
    echo "Configure rclone first:"
    echo "  $RCLONE config"
    echo ""
    exit 1
fi
echo ""

# Function to backup directory
backup_dir() {
    local src="$1"
    local dest="$2"
    local name="$3"

    if [ ! -d "$src" ]; then
        echo -e "${YELLOW}⚠${NC} $name not found at: $src"
        return
    fi

    echo "Backing up $name..."
    echo "  From: $src"
    echo "  To: ${RCLONE_REMOTE}:${BACKUP_PATH}/$dest"
    echo ""

    # Get directory size
    size=$(du -sh "$src" | cut -f1)
    echo "  Size: $size"

    # Run rclone
    $RCLONE sync "$src" "${RCLONE_REMOTE}:${BACKUP_PATH}/$dest" \
        --progress \
        --transfers 4 \
        --checkers 8 \
        --stats 10s \
        --exclude "*.pyc" \
        --exclude "__pycache__/" \
        --exclude ".git/"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $name backed up successfully"
    else
        echo -e "${RED}✗${NC} Failed to backup $name"
        return 1
    fi
    echo ""
}

# Backup data
echo "============================================================"
echo "Backing up training data..."
echo "============================================================"
echo ""

backup_dir "$DATA_DIR" "data" "Training data (data.zarr)"

# Backup checkpoints
echo "============================================================"
echo "Backing up model checkpoints..."
echo "============================================================"
echo ""

backup_dir "$CHECKPOINT_DIR" "checkpoints" "Model checkpoints"

# Backup logs
echo "============================================================"
echo "Backing up logs..."
echo "============================================================"
echo ""

if [ -d "$HOME/SalusTest/logs" ]; then
    backup_dir "$HOME/SalusTest/logs" "logs" "Training logs"
fi

# Summary
echo "============================================================"
echo "Backup Summary"
echo "============================================================"
echo ""

echo "Remote storage contents:"
$RCLONE size "${RCLONE_REMOTE}:${BACKUP_PATH}"
echo ""

echo -e "${GREEN}✅ Backup complete!${NC}"
echo ""
echo "To restore data:"
echo "  $RCLONE sync ${RCLONE_REMOTE}:${BACKUP_PATH}/data ~/salus_data_temporal"
echo "  $RCLONE sync ${RCLONE_REMOTE}:${BACKUP_PATH}/checkpoints ~/SalusTest/checkpoints"
echo ""
echo "To view backup contents:"
echo "  $RCLONE ls ${RCLONE_REMOTE}:${BACKUP_PATH}"
echo ""
