#!/bin/bash
# Test SALUS Pipeline with 1 Episode
# Run this before full deployment to verify everything works

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SALUS Pipeline Test (1 Episode)${NC}"
echo -e "${GREEN}========================================${NC}"

# Activate conda environment
echo -e "\n${YELLOW}Activating isaaclab environment...${NC}"
source ~/miniconda3/etc/profile.d/conda.sh || source ~/miniconda/etc/profile.d/conda.sh
conda activate isaaclab
echo -e "${GREEN}✓ Environment activated${NC}"

# Change to SALUS directory
cd ~/salus_a100

# Create test directory
TEST_DIR="test_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p a100_data/$TEST_DIR

echo -e "\n${YELLOW}Testing with 1 episode...${NC}"
echo -e "  Save directory: ${BLUE}a100_data/$TEST_DIR${NC}"
echo -e "  Log file: ${BLUE}a100_logs/test_run.log${NC}"
echo -e ""
echo -e "${YELLOW}This will take ~13 minutes for 1 episode${NC}"
echo -e ""

# Run collection with 1 episode
CUDA_VISIBLE_DEVICES=0 python scripts/collect_data_parallel_a100.py \
    --num_episodes 1 \
    --num_envs 1 \
    --save_dir a100_data/$TEST_DIR \
    --config configs/a100_config.yaml \
    --headless \
    --enable_cameras \
    --device cuda:0 \
    2>&1 | tee a100_logs/test_run.log

# Check if data was created
echo -e "\n${YELLOW}Verifying collected data...${NC}"

DATA_DIR=$(find a100_data/$TEST_DIR -name "data.zarr" -type d | head -n 1)

if [ -z "$DATA_DIR" ]; then
    echo -e "${RED}✗ Data collection failed - no data.zarr found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Data found: $DATA_DIR${NC}"

# Inspect data
python3 << EOF
import zarr
import sys

try:
    store = zarr.open('$DATA_DIR', mode='r')

    print("\n" + "="*60)
    print("COLLECTED DATA")
    print("="*60)
    print(f"\nDataset shapes:")
    print(f"  signals:        {store['signals'].shape}")
    print(f"  actions:        {store['actions'].shape}")
    print(f"  states:         {store['states'].shape}")
    print(f"  images:         {store['images'].shape}")
    print(f"  horizon_labels: {store['horizon_labels'].shape}")

    # Check for non-zero data
    import numpy as np
    signals_nonzero = (np.abs(store['signals'][:]) > 0).any()
    actions_nonzero = (np.abs(store['actions'][:]) > 0).any()
    images_nonzero = (store['images'][:].max() > 0)

    print(f"\nData validation:")
    print(f"  Signals populated:  {'✓' if signals_nonzero else '✗'}")
    print(f"  Actions populated:  {'✓' if actions_nonzero else '✗'}")
    print(f"  Images populated:   {'✓' if images_nonzero else '✗'}")

    if signals_nonzero and actions_nonzero and images_nonzero:
        print("\n✅ All data streams working correctly!")
        sys.exit(0)
    else:
        print("\n✗ Some data streams have issues")
        sys.exit(1)

except Exception as e:
    print(f"\n✗ Error reading data: {e}")
    sys.exit(1)
EOF

TEST_EXIT=$?

if [ $TEST_EXIT -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Pipeline Test PASSED!${NC}"
    echo -e "${GREEN}========================================${NC}"

    echo -e "\n${YELLOW}Next steps:${NC}"
    echo -e "  1. Review test log: ${BLUE}cat a100_logs/test_run.log${NC}"
    echo -e ""
    echo -e "  2. Start full data collection (500 episodes):"
    echo -e "     ${BLUE}./deploy_a100.sh collect 500 8${NC}"
    echo -e "     ${YELLOW}(This will run for ~50 hours in background)${NC}"
    echo -e ""
    echo -e "  3. Monitor progress:"
    echo -e "     ${BLUE}tail -f a100_logs/collection_500eps_*.log${NC}"

else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}✗ Pipeline Test FAILED${NC}"
    echo -e "${RED}========================================${NC}"
    echo -e "\n${YELLOW}Check the log for errors:${NC}"
    echo -e "  ${BLUE}cat a100_logs/test_run.log${NC}"
    exit 1
fi
