#!/bin/bash
#SBATCH --job-name=salus_collect
#SBATCH --output=logs/collect_%j.out
#SBATCH --error=logs/collect_%j.err
#SBATCH --time=72:00:00              # 3 days for 500 episodes
#SBATCH --partition=longq7-eng       # Long engineering partition (7 days max)
#SBATCH --gres=gpu:1                 # 1 GPU
#SBATCH --cpus-per-task=6            # Your limit: 6 cores
#SBATCH --mem=16G                    # Your limit: 16GB RAM
#SBATCH --mail-type=END,FAIL         # Email when done or failed
#SBATCH --mail-user=your_email@example.com

# SALUS Data Collection Job
# Collects 500 episodes with temporal patterns

set -e

echo "============================================================"
echo "SALUS Data Collection - SLURM Job"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Cores: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Load modules and activate conda environment
module load cuda/12.4.0
module load miniconda3/24.3.0
eval "$(conda shell.bash hook)"
conda activate isaaclab

export PYTHONPATH=$HOME/SalusTest:$PYTHONPATH

# Configuration
NUM_EPISODES=${NUM_EPISODES:-500}
NUM_ENVS=${NUM_ENVS:-2}  # Reduced to 2 (less CPU/RAM intensive)
SAVE_DIR=${SAVE_DIR:-$HOME/salus_data_temporal}

echo "Configuration:"
echo "  Episodes: $NUM_EPISODES"
echo "  Parallel envs: $NUM_ENVS"
echo "  Save dir: $SAVE_DIR"
echo ""

# Create logs directory
mkdir -p logs

# Check GPU
echo "Checking GPU..."
nvidia-smi
echo ""

# Run data collection
echo "Starting data collection..."
echo ""

cd $HOME/SalusTest

python scripts/collect_data_parallel_a100.py \
    --num_episodes $NUM_EPISODES \
    --num_envs $NUM_ENVS \
    --save_dir $SAVE_DIR \
    2>&1 | tee logs/collect_$SLURM_JOB_ID.log

echo ""
echo "============================================================"
echo "Data Collection Complete!"
echo "============================================================"
echo "End time: $(date)"
echo ""

# Check output size
if [ -d "$SAVE_DIR" ]; then
    echo "Data directory:"
    du -sh $SAVE_DIR
    ls -lh $SAVE_DIR
    echo ""
fi

# Automatically backup after collection
echo "============================================================"
echo "Starting automatic backup to cloud..."
echo "============================================================"
echo ""

if [ -f "$HOME/SalusTest/backup_from_hpc.sh" ]; then
    bash $HOME/SalusTest/backup_from_hpc.sh
    echo ""
    echo "✅ Backup complete!"
else
    echo "⚠️  Backup script not found. Backup manually with:"
    echo "  bash ~/SalusTest/backup_from_hpc.sh"
fi

echo ""
echo "============================================================"
echo "NEXT STEPS:"
echo "============================================================"
echo "1. Verify backup: ~/bin/rclone ls backup:SALUS_HPC_Backup/"
echo "2. Train model: sbatch slurm_train.sh"
echo ""
