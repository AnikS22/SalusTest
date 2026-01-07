# SALUS Paper Data Collection

## Overview

This directory contains all data collected for the SALUS paper, organized for reproducibility and analysis.

## Directory Structure

```
paper_data/
├── training_set/         # Training data (50 episodes)
├── validation_set/       # Validation data (15 episodes)
├── test_set/            # Test data (15 episodes)
├── logs/                # Collection logs and episode metadata
├── analysis/            # Data analysis scripts and results
├── checkpoints/         # Model checkpoints during training
└── figures/             # Generated figures for paper
```

## Data Collection Protocol

### Hardware Configuration
- **System**: 4× NVIDIA GeForce RTX 2080 Ti (11GB each)
- **CPU**: Intel Core i9-9820X @ 3.30GHz (10 cores, 20 threads)
- **RAM**: 64GB
- **OS**: Ubuntu 24.04.3 LTS

### Software Stack
- **Isaac Sim**: 5.1.0
- **Isaac Lab**: 0.48.5
- **VLA Model**: SmolVLA-450M (ensemble size: 1)
- **Python**: 3.11 (isaaclab conda environment)

### Collection Settings
- **Episodes per dataset**:
  - Training: 50 episodes
  - Validation: 15 episodes
  - Test: 15 episodes
- **Episode length**: 200 timesteps
- **Control frequency**: 30 Hz
- **Robot**: Franka Panda (7 DOF arm + 2 DOF gripper)
- **Cameras**: 3 RGB cameras (front, side, top views)
- **Camera resolution**: 224×224 pixels
- **Task**: Pick red cube and place in blue target zone

### Data Format (Zarr)
Each dataset contains:
- `actions`: (N, 200, 7) - VLA-generated actions (7 DOF)
- `states`: (N, 200, 7) - Robot joint positions
- `images`: (N, 200, 3, 3, 224, 224) - Camera observations (3 cameras)
- `signals`: (N, 200, 12) - Internal VLA signals (attention, uncertainty, etc.)
- `horizon_labels`: (N, 200, 4, 4) - Failure labels at 4 horizons × 4 types
- `episode_metadata`: Episode outcomes (success/failure type/length)

Compression: zstd
Chunk size: 1000 episodes

### Natural Failures

**No artificial failure injection is used.** The VLA naturally fails 20-30% of episodes through:
1. **Drop**: Cube falls off table during manipulation
2. **Timeout**: Episode exceeds 200 timesteps without success
3. **Collision**: Robot collides with objects
4. **Other**: Miscellaneous failures

This approach trains SALUS on real failure modes that occur during normal operation.

### Failure Detection
Success criterion: Cube within 5cm of goal position at episode end
Failure types automatically labeled by environment:
- **Type 0**: None (success)
- **Type 1**: Drop (cube z-position < 0.01m)
- **Type 2**: Timeout (max episode length reached)
- **Type 3**: Other (task failure)

## Reproducibility

### Running Data Collection
```bash
cd "/home/mpcr/Desktop/Salus Test/SalusTest"
conda activate isaaclab

# Training set (50 episodes)
CUDA_VISIBLE_DEVICES=0 python scripts/collect_data_franka.py \
    --num_episodes 50 \
    --save_dir paper_data/training_set/data_run1 \
    --headless \
    --enable_cameras \
    --device cuda:0

# Validation set (15 episodes)
CUDA_VISIBLE_DEVICES=0 python scripts/collect_data_franka.py \
    --num_episodes 15 \
    --save_dir paper_data/validation_set/data_run1 \
    --headless \
    --enable_cameras \
    --device cuda:0

# Test set (15 episodes)
CUDA_VISIBLE_DEVICES=0 python scripts/collect_data_franka.py \
    --num_episodes 15 \
    --save_dir paper_data/test_set/data_run1 \
    --headless \
    --enable_cameras \
    --device cuda:0
```

### Important Notes
1. **GPU Memory**: Ensure no other processes are using GPU 0 before starting collection
2. **Headless Mode**: Use `--headless --enable_cameras` for data collection (no GUI needed)
3. **Single GPU**: Run with `CUDA_VISIBLE_DEVICES=0` to avoid multi-GPU issues
4. **Kill Stale Processes**: If collection fails, check `nvidia-smi` and kill lingering Python processes

### Estimated Collection Time
- **Single episode**: ~10-15 minutes (includes VLA inference overhead)
- **50 episodes**: ~8-12 hours
- **15 episodes**: ~2.5-4 hours

### Collection Logs
All collection runs are logged in `paper_data/logs/` with:
- Standard output/error
- Episode statistics (success rate, failure types)
- Timestamps and system info

## Data Analysis

After collection, use the analysis scripts in `paper_data/analysis/` to:
- Compute success/failure rates
- Visualize episode trajectories
- Generate paper figures
- Validate data quality

## Citation

If using this data, please cite:
```bibtex
@article{salus2026,
  title={SALUS: Scalable Autonomous Learning for Uncertain Systems},
  author={[Authors]},
  journal={[Journal]},
  year={2026}
}
```

## Data Collection Status

### Training Set
- **Status**: In progress
- **Started**: January 5, 2026 06:07 UTC
- **Target episodes**: 50
- **Completed episodes**: [To be updated]
- **Success rate**: [To be computed]

### Validation Set
- **Status**: Not started
- **Target episodes**: 15

### Test Set
- **Status**: Not started
- **Target episodes**: 15

---

**Last updated**: January 5, 2026 06:10 UTC
