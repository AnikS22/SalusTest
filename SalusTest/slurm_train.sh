#!/bin/bash
#SBATCH --job-name=salus_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=8:00:00               # 8 hours (enough for 100 epochs)
#SBATCH --partition=longq7-eng       # Long partition (8h > shortq7-gpu 6h max)
#SBATCH --gres=gpu:1                 # 1 GPU
#SBATCH --cpus-per-task=6            # Your limit: 6 cores
#SBATCH --mem=16G                    # Your limit: 16GB RAM
#SBATCH --mail-type=END,FAIL         # Email when done
#SBATCH --mail-user=your_email@example.com

# SALUS Temporal Predictor Training Job

set -e

echo "============================================================"
echo "SALUS Training - SLURM Job"
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
DATA_DIR=${DATA_DIR:-$HOME/salus_data_temporal}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-64}  # Can reduce to 32 if OOM
USE_HARD_NEGATIVES=${USE_HARD_NEGATIVES:-true}
USE_FP16=${USE_FP16:-true}
SAVE_DIR=${SAVE_DIR:-$HOME/SalusTest/checkpoints/temporal_production}

echo "Configuration:"
echo "  Data dir: $DATA_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Hard negatives: $USE_HARD_NEGATIVES"
echo "  FP16: $USE_FP16"
echo "  Save dir: $SAVE_DIR"
echo ""

# Create directories
mkdir -p logs
mkdir -p $(dirname $SAVE_DIR)

# Check GPU
echo "Checking GPU..."
nvidia-smi
echo ""

# Check data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ ERROR: Data directory not found: $DATA_DIR"
    echo ""
    echo "Run data collection first:"
    echo "  sbatch slurm_collect_data.sh"
    exit 1
fi

echo "Data directory:"
du -sh $DATA_DIR
echo ""

# Run training
echo "Starting training..."
echo ""

cd $HOME/SalusTest

# Build command
CMD="python scripts/train_temporal_predictor.py \
    --data_dir $DATA_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --save_dir $SAVE_DIR"

# Add optional flags
if [ "$USE_HARD_NEGATIVES" = "true" ]; then
    CMD="$CMD --use_hard_negatives"
fi

if [ "$USE_FP16" = "true" ]; then
    CMD="$CMD --use_fp16"
fi

# Run
$CMD 2>&1 | tee logs/train_$SLURM_JOB_ID.log

echo ""
echo "============================================================"
echo "Training Complete!"
echo "============================================================"
echo "End time: $(date)"
echo ""

# Show results
if [ -d "$SAVE_DIR" ]; then
    echo "Checkpoints:"
    ls -lh $SAVE_DIR/*.pt 2>/dev/null || echo "No checkpoints found"
    echo ""

    echo "Training log:"
    tail -20 $SAVE_DIR/training.log 2>/dev/null || echo "No training log found"
    echo ""
fi

# Backup checkpoints
echo "============================================================"
echo "Backing up checkpoints..."
echo "============================================================"
echo ""

if [ -f "$HOME/SalusTest/backup_from_hpc.sh" ]; then
    bash $HOME/SalusTest/backup_from_hpc.sh
    echo "✅ Backup complete!"
else
    echo "⚠️  Backup manually with: bash ~/SalusTest/backup_from_hpc.sh"
fi

echo ""
echo "============================================================"
echo "RESULTS SUMMARY"
echo "============================================================"
echo ""
echo "View training curves:"
echo "  tensorboard --logdir $SAVE_DIR/logs_*"
echo ""
echo "Best model:"
echo "  $SAVE_DIR/best_model.pt"
echo ""
