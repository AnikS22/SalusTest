#!/bin/bash
# Quick sync to Athene - Run this manually

export HPC_HOST=athene-login.hpc.fau.edu
export HPC_USER=asahai2024
export HPC_PATH=SalusTest  # Relative to home, no tilde

echo "============================================================"
echo "Syncing SALUS to Athene"
echo "============================================================"
echo ""
echo "User: asahai2024@athene-login.hpc.fau.edu"
echo "Path: ~/SalusTest"
echo ""

# Test SSH (will prompt for password if no key)
echo "Testing connection (you may need to enter password)..."
ssh ${HPC_USER}@${HPC_HOST} "mkdir -p \$HOME/${HPC_PATH} && echo 'Directory ready'"

if [ $? -ne 0 ]; then
    echo ""
    echo "Cannot connect to Athene."
    echo "Please make sure you can SSH manually first:"
    echo "  ssh asahai2024@athene-login.hpc.fau.edu"
    exit 1
fi

echo ""
echo "Connection OK! Starting sync..."
echo ""

# Sync with rsync
rsync -avz --progress \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='venv*/' \
    --exclude='data/' \
    --exclude='checkpoints/' \
    --exclude='logs/' \
    --exclude='*.zarr/' \
    --exclude='paper_data/data.zarr/' \
    --exclude='paper_data/logs/' \
    --exclude='models/tinyvla/' \
    --exclude='models/openvla/' \
    ./ ${HPC_USER}@${HPC_HOST}:${HPC_PATH}/

echo ""
echo "============================================================"
echo "✅ Sync Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. SSH: ssh asahai2024@athene-login.hpc.fau.edu"
echo "  2. Navigate: cd ~/SalusTest"
echo "  3. Test: python scripts/test_hpc_phase1.py"
echo ""
