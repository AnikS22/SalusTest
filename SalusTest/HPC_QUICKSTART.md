# SALUS HPC Quick Start Guide

**Status**: ✅ All local tests passing, ready for HPC deployment

## Pre-Flight Checklist

### ✅ Local Tests (COMPLETED)
- [x] All 4 Phase 1 tests passed
- [x] Component tests: 7/7 passing
- [x] Quick proof test: 97.5% pattern discrimination
- [x] Repository cleaned and organized

### 📋 HPC Setup (TODO)
- [ ] SSH access configured
- [ ] Repository synced to HPC
- [ ] Phase 1 tests pass on HPC
- [ ] Small-scale data collection (50 episodes)
- [ ] Full production training (500 episodes)

---

## Step 1: Configure SSH Access to HPC

### Setup SSH Key (if not already done)

```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy public key to HPC
ssh-copy-id your_username@athene.informatik.tu-darmstadt.de

# Test connection
ssh your_username@athene.informatik.tu-darmstadt.de
```

### Test Connection

```bash
# Should connect without password
ssh your_username@athene.informatik.tu-darmstadt.de "echo 'Connection OK'"
```

---

## Step 2: Sync Repository to HPC

### Set Environment Variables

```bash
# Export HPC configuration
export HPC_HOST=athene.informatik.tu-darmstadt.de
export HPC_USER=your_username
export HPC_PATH=~/SalusTest  # Or your preferred path
```

### Run Sync Script

```bash
cd /home/mpcr/Desktop/Salus\ Test/SalusTest

# Sync to HPC
./sync_to_hpc.sh
```

**What gets synced**:
- ✅ All Python code (`salus/`, `scripts/`)
- ✅ Documentation (`docs/`, `*.md`)
- ✅ Configuration files
- ❌ Data directories (too large)
- ❌ Model checkpoints (download separately on HPC)
- ❌ Python cache files
- ❌ Git history

**Sync time**: ~30 seconds (first time), ~5 seconds (updates)

---

## Step 3: Run Tests on HPC

### SSH to HPC

```bash
ssh your_username@athene.informatik.tu-darmstadt.de
```

### Navigate to Project

```bash
cd ~/SalusTest
```

### Run Phase 1 Validation

```bash
# This will test:
# 1. Python/CUDA
# 2. Imports
# 3. Component tests (7 tests)
# 4. Quick proof test

python scripts/test_hpc_phase1.py
```

**Expected output**:
```
============================================================
PHASE 1 VALIDATION SUMMARY
============================================================

Tests passed: 4/4

✅ ALL TESTS PASSED!
```

**If tests fail**:
- Check Python version: `python --version` (need 3.9+)
- Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU: `nvidia-smi`

---

## Step 4: Phase 2 - Small Scale Test (30 min)

### Collect 50 Episodes

```bash
# On HPC
cd ~/SalusTest

python scripts/collect_data_parallel_a100.py \
    --num_episodes 50 \
    --num_envs 2 \
    --save_dir ~/salus_test_data
```

**Expected**:
- Runtime: ~15-20 minutes
- Output: `~/salus_test_data/data.zarr/`
- Size: ~500MB-1GB

### Train on Test Data

```bash
python scripts/train_temporal_predictor.py \
    --data_dir ~/salus_test_data \
    --epochs 20 \
    --batch_size 32 \
    --save_dir checkpoints/hpc_test
```

**Expected**:
- Runtime: ~5-10 minutes
- Training loss should decrease
- Validation F1 > 0.4

### Check Results

```bash
# View training logs
cat checkpoints/hpc_test/training.log

# Check if model was saved
ls -lh checkpoints/hpc_test/best_model.pt
```

**Success criteria**:
- ✅ Data collection completes
- ✅ Training converges (loss decreases)
- ✅ Model checkpoint saved
- ✅ No CUDA errors

---

## Step 5: Phase 3 - Full Production (48+ hours)

### Collect 500 Episodes

```bash
# On HPC
cd ~/SalusTest

# Start data collection (will take ~40-50 hours)
python scripts/collect_data_parallel_a100.py \
    --num_episodes 500 \
    --num_envs 4 \
    --save_dir ~/salus_data_temporal

# Monitor progress (in another terminal)
watch -n 60 'ls -lh ~/salus_data_temporal/data.zarr/'
```

**Expected**:
- Runtime: ~40-50 hours
- Output size: ~10-15GB
- GPU utilization: 80-95%

### Train Production Model

```bash
# Basic training
python scripts/train_temporal_predictor.py \
    --data_dir ~/salus_data_temporal \
    --epochs 100 \
    --batch_size 64 \
    --use_hard_negatives \
    --save_dir checkpoints/temporal_baseline

# Advanced training with latent compression
python scripts/train_temporal_predictor.py \
    --data_dir ~/salus_data_temporal \
    --use_latent_encoder \
    --latent_dim 6 \
    --epochs 100 \
    --use_fp16 \
    --save_dir checkpoints/temporal_latent
```

**Expected**:
- Runtime: ~2-4 hours on A100
- Target: **F1 > 0.60** (2× baseline)
- Final model size: ~200KB

### Monitor Training

```bash
# In another terminal on HPC
ssh your_username@athene.informatik.tu-darmstadt.de
cd ~/SalusTest

# View real-time training logs
tail -f checkpoints/temporal_baseline/training.log

# Or use TensorBoard
tensorboard --logdir checkpoints/temporal_baseline/logs_*
```

---

## Quick Command Reference

### Local Machine

```bash
# Run all local tests
cd /home/mpcr/Desktop/Salus\ Test/SalusTest
python scripts/test_hpc_phase1.py

# Sync to HPC
export HPC_HOST=athene.informatik.tu-darmstadt.de
export HPC_USER=your_username
./sync_to_hpc.sh

# Update HPC after local changes
./sync_to_hpc.sh  # Run again to sync updates
```

### HPC

```bash
# Connect
ssh your_username@athene.informatik.tu-darmstadt.de

# Run tests
cd ~/SalusTest
python scripts/test_hpc_phase1.py

# Small test (Phase 2)
python scripts/collect_data_parallel_a100.py --num_episodes 50 --num_envs 2
python scripts/train_temporal_predictor.py --data_dir ~/salus_test_data --epochs 20

# Full production (Phase 3)
python scripts/collect_data_parallel_a100.py --num_episodes 500 --num_envs 4
python scripts/train_temporal_predictor.py --data_dir ~/salus_data_temporal --epochs 100
```

---

## Troubleshooting

### Sync Issues

**Problem**: `Connection refused`
```bash
# Test SSH
ssh your_username@athene.informatik.tu-darmstadt.de

# If that fails, check SSH key
cat ~/.ssh/id_ed25519.pub
ssh-copy-id your_username@athene.informatik.tu-darmstadt.de
```

**Problem**: `Permission denied`
```bash
# Make sure HPC_PATH exists
ssh your_username@athene.informatik.tu-darmstadt.de "mkdir -p ~/SalusTest"
```

### HPC Test Failures

**Problem**: `No module named 'torch'`
```bash
# Install dependencies on HPC
pip install torch torchvision numpy zarr tqdm tensorboard scikit-learn matplotlib
```

**Problem**: `CUDA not available`
```bash
# Check GPU
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If false, reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Problem**: `ImportError: cannot import name 'HybridTemporalPredictor'`
```bash
# Check files synced correctly
ls -la ~/SalusTest/salus/models/

# Re-sync if needed
# (from local machine) ./sync_to_hpc.sh
```

### Data Collection Issues

**Problem**: `Isaac Lab not found`
```bash
# Check Isaac Lab installation on HPC
python -c "import isaaclab; print(isaaclab.__version__)"

# If not installed, see docs/ISAACLAB_SETUP.md
```

**Problem**: `CUDA out of memory`
```bash
# Reduce parallel environments
python scripts/collect_data_parallel_a100.py --num_episodes 500 --num_envs 2

# Or use smaller batch size during training
python scripts/train_temporal_predictor.py --batch_size 32
```

---

## Performance Monitoring

### Check GPU Usage

```bash
# On HPC
watch -n 1 nvidia-smi
```

### Check Disk Space

```bash
# Check available space
df -h ~/

# Check data directory size
du -sh ~/salus_data_temporal/
```

### Monitor Training Progress

```bash
# Training logs
tail -f checkpoints/temporal_baseline/training.log

# Checkpoints
ls -lh checkpoints/temporal_baseline/*.pt

# TensorBoard (requires port forwarding)
tensorboard --logdir checkpoints/temporal_baseline/logs_*
```

---

## Expected Timeline

| Phase | Task | Duration | Notes |
|-------|------|----------|-------|
| **Phase 1** | Local tests | 5-10 min | ✅ Done |
| **Phase 1** | Sync to HPC | 30 sec | Ready to run |
| **Phase 1** | HPC tests | 5-10 min | Next step |
| **Phase 2** | Collect 50 episodes | 15-20 min | Small test |
| **Phase 2** | Train test model | 5-10 min | Validate pipeline |
| **Phase 3** | Collect 500 episodes | 40-50 hours | Production data |
| **Phase 3** | Train production | 2-4 hours | Final model |

**Total**: ~2-3 days (mostly data collection time)

---

## Success Criteria

### Phase 1 ✅
- [x] Local tests: 4/4 passing
- [x] Sync script created
- [ ] HPC tests: 4/4 passing

### Phase 2
- [ ] 50 episodes collected successfully
- [ ] Training converges (loss decreases)
- [ ] No CUDA errors
- [ ] Validation F1 > 0.4

### Phase 3 (Production)
- [ ] 500 episodes collected
- [ ] Training completes (100 epochs)
- [ ] **Target: F1 > 0.60** (2× baseline)
- [ ] Model checkpoints saved
- [ ] TensorBoard logs generated

---

## Next Actions

1. **Now**: Set HPC credentials
   ```bash
   export HPC_HOST=athene.informatik.tu-darmstadt.de
   export HPC_USER=your_username
   ```

2. **Now**: Run sync script
   ```bash
   ./sync_to_hpc.sh
   ```

3. **Then**: SSH to HPC and run tests
   ```bash
   ssh your_username@athene.informatik.tu-darmstadt.de
   cd ~/SalusTest
   python scripts/test_hpc_phase1.py
   ```

4. **If tests pass**: Start Phase 2
   ```bash
   python scripts/collect_data_parallel_a100.py --num_episodes 50 --num_envs 2
   ```

---

**Last Updated**: January 6, 2026
**Status**: ✅ Ready for HPC deployment
**Next Step**: Configure SSH and run sync script
