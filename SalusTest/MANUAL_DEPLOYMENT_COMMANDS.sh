#!/bin/bash
# Manual Deployment Commands for Athene HPC
# Copy-paste these commands one by one

echo "============================================"
echo "SALUS Manual Deployment to Athene HPC"
echo "============================================"

echo ""
echo "Step 1: Transfer deployment package"
echo "Run this on your LOCAL machine:"
echo ""
echo "cd \"/home/mpcr/Desktop/Salus Test/SalusTest\""
echo "scp salus_a100_deployment_20260105_200242.tar.gz asahai2024@athene-login.hpc.fau.edu:~/"
echo ""
echo "Press Enter after transfer completes..."
read

echo ""
echo "Step 2: SSH into Athene"
echo "Run this:"
echo ""
echo "ssh asahai2024@athene-login.hpc.fau.edu"
echo ""
echo "(You are now on Athene HPC login node)"
echo ""

echo "Step 3: Extract package"
echo "Run these commands on Athene:"
echo ""
cat << 'EOF'
tar -xzf salus_a100_deployment_20260105_200242.tar.gz
cd salus_a100_deployment_20260105_200242
ls -la
EOF

echo ""
echo "Step 4: Request GPU node"
echo "Run this on Athene:"
echo ""
cat << 'EOF'
# Try this first (adjust partition name if needed)
srun --partition=gpu-a100 --gpus=1 --ntasks=8 --mem=64G --time=02:00:00 --pty bash

# If above fails, check available partitions:
# sinfo

# Then try with correct partition name:
# srun --partition=<correct-partition> --gpus=1 --time=02:00:00 --pty bash
EOF

echo ""
echo "Step 5: Verify GPU access"
echo "Run this on the GPU node:"
echo ""
echo "nvidia-smi"
echo ""
echo "(Should show A100 GPU)"
echo ""

echo "Step 6: Run setup"
echo "Run this on the GPU node:"
echo ""
echo "bash hpc_setup.sh"
echo ""
echo "(Takes ~15 minutes, downloads models)"
echo ""

echo "Step 7: Configure Rclone backup"
echo "Run this:"
echo ""
echo "bash setup_rclone.sh"
echo ""
echo "(Follow interactive prompts)"
echo ""

echo "Step 8: Test with 1 episode"
echo "Run this:"
echo ""
echo "bash test_salus.sh"
echo ""
echo "(Takes ~13 minutes)"
echo ""
echo "If test passes, continue to Step 9"
echo ""

echo "Step 9: Submit full collection job"
echo "Run these commands:"
echo ""
cat << 'EOF'
# Exit GPU node (Ctrl+D)
# Return to login node

# Submit batch job
sbatch submit_salus.sh

# Check job status
squeue -u asahai2024

# Monitor progress
tail -f salus_*.out
EOF

echo ""
echo "============================================"
echo "Deployment will run for ~50 hours"
echo "Monitor with: squeue -u asahai2024"
echo "View logs: tail -f salus_*.out"
echo "============================================"
