# SALUS HPC Sync - Exact Details

## What Gets Synced to HPC

### ✅ Code Only (~2MB)

```
SalusTest/
├── salus/                    408KB    ← All Python modules
│   ├── models/
│   │   ├── temporal_predictor.py
│   │   ├── latent_encoder.py
│   │   └── failure_predictor.py
│   ├── data/
│   │   ├── temporal_dataset.py
│   │   └── preprocess_labels.py
│   └── simulation/
│
├── scripts/                  228KB    ← All executable scripts
│   ├── test_hpc_phase1.py
│   ├── collect_data_parallel_a100.py
│   ├── train_temporal_predictor.py
│   └── [15+ other scripts]
│
├── docs/                     832KB    ← All documentation
│   ├── TEMPORAL_IMPLEMENTATION_SUMMARY.md
│   ├── GETTING_STARTED.md
│   └── [30+ other docs]
│
├── *.md                      ~200KB   ← Root documentation
├── *.sh                      ~50KB    ← Setup/backup scripts
├── slurm_*.sh               ~20KB    ← SLURM job scripts
└── configs/                  16KB     ← Configuration files

TOTAL: ~2MB
```

### ❌ What's EXCLUDED (Automatic)

```
NOT synced:
├── .git/                     ← Git history (large)
├── venv_salus/              5.9GB  ← Virtual environment
├── data/                    ~10GB  ← Training data (too big)
├── checkpoints/             ~1GB   ← Model checkpoints
├── logs/                    ~100MB ← Log files
├── paper_data/              ~15GB  ← Old data
├── __pycache__/             ← Python cache
├── *.pyc                    ← Compiled Python
├── *.zarr/                  ← Data directories
└── models/tinyvla/          ← Downloaded models
```

**Sync time**: ~5-10 seconds (only 2MB!)

---

## Resource Requirements

### Your HPC Limits

- **CPUs**: 6 cores max ✅
- **RAM**: 16GB max ✅
- **GPU**: 1 GPU ✅

### What We Actually Need

| Job | CPUs | RAM | GPU | Time | Notes |
|-----|------|-----|-----|------|-------|
| **Test (Phase 1)** | 2 | 8GB | 1 | 30 min | Light testing |
| **Data Collection** | 6 | 16GB | 1 | 48-72h | Parallel envs |
| **Training** | 4 | 12GB | 1 | 4-8h | GPU-bound |

**All jobs fit within your limits!** ✅

### Why 6 Cores is Fine

- **Data collection**: GPU bottleneck (not CPU)
- **Training**: GPU bottleneck (batch processing)
- **6 cores**: Enough for data loading + GPU feeds

We reduced `num_envs` from 4 → 2 to fit in your limits.

---

## SLURM Jobs Created

### 1. Test Job (Quick Validation)

**File**: `slurm_test_phase1.sh`

```bash
#SBATCH --cpus-per-task=2      # Light: only 2 cores
#SBATCH --mem=8G               # Light: only 8GB
#SBATCH --time=00:30:00        # 30 minutes
```

**Purpose**: Verify everything works
**Runtime**: 5-10 minutes
**Run first**: `sbatch slurm_test_phase1.sh`

### 2. Data Collection Job

**File**: `slurm_collect_data.sh`

```bash
#SBATCH --cpus-per-task=6      # Your max: 6 cores
#SBATCH --mem=16G              # Your max: 16GB
#SBATCH --time=72:00:00        # 3 days
```

**Purpose**: Collect 500 episodes
**Runtime**: 40-50 hours
**Output**: ~10-15GB in `~/salus_data_temporal/`
**Auto-backup**: Yes, backs up to cloud after completion

**Resource usage**:
- GPU: 80-95% (simulation rendering)
- CPU: 40-60% (2-3 cores active)
- RAM: 8-12GB (well within 16GB)

### 3. Training Job

**File**: `slurm_train.sh`

```bash
#SBATCH --cpus-per-task=6      # Your max: 6 cores
#SBATCH --mem=16G              # Your max: 16GB
#SBATCH --time=08:00:00        # 8 hours
```

**Purpose**: Train temporal predictor
**Runtime**: 2-4 hours (100 epochs)
**Output**: ~200KB model checkpoint
**Auto-backup**: Yes, backs up checkpoints

**Resource usage**:
- GPU: 90-99% (training)
- CPU: 20-40% (data loading)
- RAM: 4-8GB (batch size 64)

---

## Complete Workflow

### Step 1: Sync Code to HPC (30 seconds)

```bash
# On LOCAL machine
export HPC_HOST=athene.informatik.tu-darmstadt.de
export HPC_USER=your_username

cd /home/mpcr/Desktop/Salus\ Test/SalusTest
./sync_to_hpc.sh
```

**Uploads**: ~2MB code only
**Time**: 5-10 seconds

### Step 2: Setup HPC Environment (5 minutes, once)

```bash
# SSH to HPC
ssh your_username@athene.informatik.tu-darmstadt.de
cd ~/SalusTest

# Make scripts executable
chmod +x slurm_*.sh *.sh

# Create logs directory
mkdir -p logs

# Setup rclone for backup
bash setup_rclone_hpc.sh
~/bin/rclone config  # Add Google Drive as 'backup'

# Test
~/bin/rclone lsd backup:
```

### Step 3: Run Phase 1 Tests (10 minutes)

```bash
# On HPC
cd ~/SalusTest

# Option A: Interactive test (if you have interactive GPU access)
python scripts/test_hpc_phase1.py

# Option B: Submit SLURM job (recommended)
sbatch slurm_test_phase1.sh

# Check job status
squeue -u $USER

# View results when done
cat logs/test_*.out
```

**Expected output**: 4/4 tests passed

### Step 4: Start Data Collection (48-72 hours)

```bash
# On HPC
cd ~/SalusTest

# Submit collection job
sbatch slurm_collect_data.sh

# Check status
squeue -u $USER

# Monitor progress (in real-time)
tail -f logs/collect_*.out

# Or check periodically
watch -n 300 'ls -lh ~/salus_data_temporal/; squeue -u $USER'
```

**The job will**:
1. Collect 500 episodes (~48 hours)
2. Automatically backup to cloud (~30 min)
3. Email you when complete

### Step 5: Train Model (4-8 hours)

```bash
# On HPC (after data collection completes)
cd ~/SalusTest

# Submit training job
sbatch slurm_train.sh

# Monitor
tail -f logs/train_*.out

# Check GPU usage
squeue -u $USER
```

**The job will**:
1. Train for 100 epochs (~4 hours)
2. Save best model checkpoint
3. Automatically backup checkpoints
4. Email you when complete

### Step 6: Download Results (Optional, 30-60 min)

```bash
# On LOCAL machine
export HPC_HOST=athene.informatik.tu-darmstadt.de
export HPC_USER=your_username

cd /home/mpcr/Desktop/Salus\ Test/SalusTest
./sync_from_hpc.sh
```

**Downloads**:
- Training data: ~10-15GB → `~/salus_data_hpc/`
- Checkpoints: ~1MB → `~/SalusTest/checkpoints_hpc/`

---

## Resource Monitoring

### Check Your Limits

```bash
# On HPC
# Check CPU/RAM limits
sinfo -o "%P %l %c %m"

# Check your quota
quota -s

# Check running jobs
squeue -u $USER

# Check job details
scontrol show job JOBID
```

### Monitor GPU Usage

```bash
# During job execution (if you have access to node)
ssh compute-node-XX
watch -n 1 nvidia-smi
```

### Disk Space

```bash
# Check available space
df -h ~/

# Check data size
du -sh ~/salus_data_temporal/
du -sh ~/SalusTest/checkpoints/

# Monitor during collection
watch -n 60 'du -sh ~/salus_data_temporal/'
```

---

## Job Management

### Submit Jobs

```bash
# Test
sbatch slurm_test_phase1.sh

# Data collection
sbatch slurm_collect_data.sh

# Training
sbatch slurm_train.sh
```

### Check Status

```bash
# Your jobs
squeue -u $USER

# Job details
scontrol show job JOBID

# Cancel job
scancel JOBID
```

### View Logs

```bash
# Test logs
cat logs/test_*.out
tail logs/test_*.err

# Collection logs
tail -f logs/collect_*.out

# Training logs
tail -f logs/train_*.out
```

---

## Customization

### Reduce Resource Usage Further

If you need even less resources:

**Data Collection**:
```bash
# Edit slurm_collect_data.sh
#SBATCH --cpus-per-task=4      # Reduce to 4 cores
#SBATCH --mem=12G              # Reduce to 12GB

# Reduce parallel environments
NUM_ENVS=1  # Only 1 environment (slower but uses less RAM)
```

**Training**:
```bash
# Edit slurm_train.sh
#SBATCH --cpus-per-task=4      # Reduce to 4 cores
#SBATCH --mem=12G              # Reduce to 12GB

# Reduce batch size
BATCH_SIZE=32  # Smaller batches (uses less GPU RAM)
```

### Faster Testing (Phase 2 Small Test)

```bash
# Collect only 50 episodes for quick test
sbatch --export=NUM_EPISODES=50,NUM_ENVS=1 slurm_collect_data.sh

# Quick training
sbatch --export=EPOCHS=20,BATCH_SIZE=32 slurm_train.sh
```

---

## Email Notifications

Edit SLURM scripts to add your email:

```bash
# In each slurm_*.sh file
#SBATCH --mail-user=your_email@example.com  # ← Change this!
```

You'll get emails for:
- ✅ Job completed
- ❌ Job failed
- ⏰ Job timeout

---

## Estimated Timeline

| Step | Duration | Waiting | Notes |
|------|----------|---------|-------|
| **Sync code** | 10 sec | Active | Upload 2MB |
| **Setup HPC** | 5 min | Active | One-time setup |
| **Test** | 10 min | SLURM queue | Validate system |
| **Data collection** | 48 hours | SLURM queue | Main wait |
| **Backup** | 30 min | Auto | After collection |
| **Training** | 4 hours | SLURM queue | Fast |
| **Download** | 30 min | Optional | If needed |

**Total**: ~2-3 days (mostly waiting for SLURM jobs)

---

## Troubleshooting

### "Job pending forever"

```bash
# Check queue
squeue -u $USER

# Check why pending
squeue -j JOBID --start

# Reason might be:
# - QOSMaxCpuPerUserLimit: You have other jobs running
# - Resources: Cluster busy
# - Priority: Low priority
```

### "Out of memory"

```bash
# Reduce resources in SLURM script
#SBATCH --mem=12G  # Instead of 16G

# Or reduce batch size
BATCH_SIZE=32  # Instead of 64
```

### "GPU not available"

```bash
# Check GPU partition
sinfo -p gpu

# Make sure you requested GPU
#SBATCH --gres=gpu:1
```

---

## Quick Reference

| File | Purpose | When to Run |
|------|---------|-------------|
| `sync_to_hpc.sh` | Upload code | First + updates |
| `setup_rclone_hpc.sh` | Setup backup | Once |
| `slurm_test_phase1.sh` | Validate | First |
| `slurm_collect_data.sh` | Collect data | After test passes |
| `slurm_train.sh` | Train model | After data collected |
| `backup_from_hpc.sh` | Manual backup | If auto-backup fails |
| `sync_from_hpc.sh` | Download results | Optional |

---

**Summary**: You're sending only ~2MB of code, and all jobs fit within your 6 core / 16GB limits! ✅
