#!/bin/bash
#SBATCH --job-name=salus_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --time=00:30:00              # 30 minutes
#SBATCH --partition=shortq7-gpu      # GPU partition (6h max)
#SBATCH --gres=gpu:1                 # 1 GPU
#SBATCH --cpus-per-task=2            # Only 2 cores needed for tests
#SBATCH --mem=8G                     # Only 8GB needed for tests
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@example.com

# SALUS Phase 1 Validation Tests

set -e

echo "============================================================"
echo "SALUS Phase 1 Validation - SLURM Job"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load modules and activate conda environment
module load cuda/12.4.0
module load miniconda3/24.3.0
eval "$(conda shell.bash hook)"
conda activate isaaclab

export PYTHONPATH=$HOME/SalusTest:$PYTHONPATH

mkdir -p logs

# Check environment
echo "Checking Python..."
python --version
echo ""

echo "Checking PyTorch..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
echo ""

echo "Checking GPU..."
nvidia-smi
echo ""

# Run Phase 1 tests
echo "Running Phase 1 validation tests..."
echo ""

cd $HOME/SalusTest
python scripts/test_hpc_phase1.py 2>&1 | tee logs/phase1_test_$SLURM_JOB_ID.log

TEST_RESULT=$?

echo ""
echo "============================================================"
echo "Test Results"
echo "============================================================"
echo ""

if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ ALL TESTS PASSED!"
    echo ""
    echo "System is ready for data collection."
    echo ""
    echo "Next steps:"
    echo "  1. Setup rclone: bash setup_rclone_hpc.sh"
    echo "  2. Start collection: sbatch slurm_collect_data.sh"
    echo ""
else
    echo "❌ SOME TESTS FAILED"
    echo ""
    echo "Check logs: logs/test_$SLURM_JOB_ID.out"
    echo ""
fi

echo "End time: $(date)"
