#!/bin/bash
# SALUS A100 Deployment Script
# Automated deployment for A100 data collection and training

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SALUS A100 Deployment${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if on A100
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo -e "\n${YELLOW}Detected GPU:${NC} $GPU_NAME"

if [[ ! $GPU_NAME =~ "A100" ]]; then
    echo -e "${RED}WARNING: This script is optimized for NVIDIA A100${NC}"
    echo -e "${RED}Detected GPU: $GPU_NAME${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
ACTION=${1:-"collect"}  # collect, train, or both
NUM_EPISODES=${2:-500}
NUM_ENVS=${3:-8}

echo -e "\n${YELLOW}Configuration:${NC}"
echo "  Action: $ACTION"
echo "  Episodes: $NUM_EPISODES"
echo "  Parallel Envs: $NUM_ENVS"

# Create directories
mkdir -p a100_data
mkdir -p a100_checkpoints
mkdir -p a100_logs
mkdir -p a100_results

# Activate conda environment
echo -e "\n${YELLOW}Activating conda environment...${NC}"
source ~/miniconda/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab
echo -e "${GREEN}✓ Environment activated${NC}"

# Function to collect data
collect_data() {
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Starting Data Collection${NC}"
    echo -e "${GREEN}========================================${NC}"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="a100_logs/collection_${NUM_EPISODES}eps_${TIMESTAMP}.log"

    echo "  Target: $NUM_EPISODES episodes"
    echo "  Parallel envs: $NUM_ENVS"
    echo "  Log: $LOG_FILE"

    CUDA_VISIBLE_DEVICES=0 python scripts/collect_data_parallel_a100.py \
        --num_episodes $NUM_EPISODES \
        --num_envs $NUM_ENVS \
        --save_dir a100_data/training_${NUM_EPISODES}eps \
        --config configs/a100_config.yaml \
        --headless \
        --enable_cameras \
        --device cuda:0 \
        > "$LOG_FILE" 2>&1 &

    COLLECT_PID=$!
    echo -e "${GREEN}✓ Collection started (PID: $COLLECT_PID)${NC}"
    echo "  Monitor: tail -f $LOG_FILE"

    # Wait for collection to complete
    echo -e "\n${YELLOW}Waiting for collection to complete...${NC}"
    echo "  (This will take ~50-100 hours for 500 episodes)"
    echo "  Press Ctrl+C to stop monitoring (collection continues in background)"

    # Monitor progress
    while kill -0 $COLLECT_PID 2>/dev/null; do
        sleep 60
        PROGRESS=$(grep -oP "Progress: \K\d+/\d+" "$LOG_FILE" | tail -n 1 || echo "0/$NUM_EPISODES")
        echo "  Progress: $PROGRESS episodes"
    done

    wait $COLLECT_PID
    COLLECT_EXIT=$?

    if [ $COLLECT_EXIT -eq 0 ]; then
        echo -e "${GREEN}✓ Data collection complete!${NC}"
        return 0
    else
        echo -e "${RED}✗ Data collection failed with exit code $COLLECT_EXIT${NC}"
        echo "  Check log: $LOG_FILE"
        return 1
    fi
}

# Function to train model
train_model() {
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Starting Model Training${NC}"
    echo -e "${GREEN}========================================${NC}"

    # Find most recent data directory
    DATA_DIR=$(ls -td a100_data/training_${NUM_EPISODES}eps/*/ 2>/dev/null | head -n 1)

    if [ -z "$DATA_DIR" ]; then
        echo -e "${RED}✗ No data found in a100_data/training_${NUM_EPISODES}eps/${NC}"
        echo "  Run collection first or specify data path"
        return 1
    fi

    DATA_PATH="${DATA_DIR}data.zarr"
    echo "  Data: $DATA_PATH"

    # Check if data exists
    if [ ! -d "$DATA_PATH" ]; then
        echo -e "${RED}✗ Data not found: $DATA_PATH${NC}"
        return 1
    fi

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="a100_logs/training_${NUM_EPISODES}eps_${TIMESTAMP}.log"

    echo "  Model: large (200K params)"
    echo "  Batch size: 1024"
    echo "  Epochs: 100"
    echo "  Log: $LOG_FILE"

    python scripts/train_failure_predictor_a100.py \
        --data_path "$DATA_PATH" \
        --save_dir a100_checkpoints \
        --batch_size 1024 \
        --num_epochs 100 \
        --model_size large \
        --use_amp \
        --num_workers 8 \
        > "$LOG_FILE" 2>&1

    TRAIN_EXIT=$?

    if [ $TRAIN_EXIT -eq 0 ]; then
        echo -e "${GREEN}✓ Training complete!${NC}"

        # Display results
        if [ -f "a100_checkpoints/training_results_a100.json" ]; then
            echo -e "\n${YELLOW}Results:${NC}"
            python3 << 'EOF'
import json
with open('a100_checkpoints/training_results_a100.json') as f:
    results = json.load(f)
    metrics = results['test_metrics']
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
EOF
        fi
        return 0
    else
        echo -e "${RED}✗ Training failed with exit code $TRAIN_EXIT${NC}"
        echo "  Check log: $LOG_FILE"
        return 1
    fi
}

# Main execution
case $ACTION in
    collect)
        collect_data
        ;;
    train)
        train_model
        ;;
    both)
        if collect_data; then
            train_model
        else
            echo -e "${RED}✗ Skipping training due to collection failure${NC}"
            exit 1
        fi
        ;;
    *)
        echo -e "${RED}Unknown action: $ACTION${NC}"
        echo "Usage: $0 [collect|train|both] [num_episodes] [num_envs]"
        echo "Examples:"
        echo "  $0 collect 500 8      # Collect 500 episodes with 8 parallel envs"
        echo "  $0 train              # Train on most recent data"
        echo "  $0 both 1000 8        # Collect 1000 episodes then train"
        exit 1
        ;;
esac

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}SALUS A100 Deployment Complete${NC}"
echo -e "${GREEN}========================================${NC}"
