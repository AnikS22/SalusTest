# SALUS HPC Launch Checklist

**Date**: January 6, 2026
**Status**: ✅ READY TO LAUNCH
**Your Limits**: 6 cores, 16GB RAM, 1 GPU ✅ ALL JOBS FIT!

---

## 📦 What You're Sending

```
ONLY CODE (~2MB):
├── salus/          408KB   ← Python modules
├── scripts/        228KB   ← Scripts
├── docs/           832KB   ← Documentation
├── *.md            200KB   ← Guides
├── slurm_*.sh      20KB    ← SLURM jobs
└── *.sh            50KB    ← Backup scripts

EXCLUDED (Automatic):
├── venv_salus/     5.9GB   ← Not synced
├── data/           10GB    ← Not synced
├── .git/                   ← Not synced
└── __pycache__/            ← Not synced

TOTAL UPLOAD: ~2MB (5-10 seconds)
```

---

## 🚀 Launch Commands (Copy-Paste Ready!)

### 1. Configure HPC (First Time Only)

```bash
# Set these variables (EDIT YOUR USERNAME!)
export HPC_HOST=athene-login.hpc.fau.edu
export HPC_USER=your_username  # ← CHANGE THIS!
export HPC_PATH=~/SalusTest
```

### 2. Sync Code to HPC

```bash
cd /home/mpcr/Desktop/Salus\ Test/SalusTest

# This uploads ONLY 2MB of code (fast!)
./sync_to_hpc.sh
```

**Expected output**:
```
✅ Sync completed successfully!

Next steps:
  1. SSH to HPC: ssh your_username@athene-login.hpc.fau.edu
  2. Navigate: cd ~/SalusTest
  3. Run tests: python scripts/test_hpc_phase1.py
```

### 3. Setup HPC Environment (One Time)

```bash
# SSH to HPC
ssh your_username@athene-login.hpc.fau.edu

# Navigate to project
cd ~/SalusTest

# Make scripts executable
chmod +x slurm_*.sh *.sh

# Create logs directory
mkdir -p logs

# Setup rclone for data backup
bash setup_rclone_hpc.sh

# Configure Google Drive (or other cloud)
~/bin/rclone config
# n) New remote
# name> backup
# Storage> drive  (for Google Drive)
# [Follow authorization prompts]

# Test connection
~/bin/rclone lsd backup:
# Should show: (no error = success!)
```

### 4. Run Phase 1 Tests

```bash
# On HPC
cd ~/SalusTest

# Submit test job
sbatch slurm_test_phase1.sh

# Check status
squeue -u $USER

# Wait ~10 minutes, then check results
cat logs/test_*.out
```

**Success looks like**:
```
✅ ALL TESTS PASSED!
Tests passed: 4/4
```

### 5. Start Data Collection (48+ hours)

```bash
# On HPC (after tests pass)
cd ~/SalusTest

# Submit collection job
sbatch slurm_collect_data.sh

# Monitor (optional)
tail -f logs/collect_*.out

# Or check periodically
squeue -u $USER
```

**The job will**:
- Collect 500 episodes (~48 hours)
- Use 6 cores, 16GB RAM, 1 GPU (within your limits!)
- Output ~10-15GB to `~/salus_data_temporal/`
- **Automatically backup to cloud** (~30 min)
- Email you when complete

### 6. Train Model (4-8 hours)

```bash
# On HPC (after data collection completes)
cd ~/SalusTest

# Check collection completed
ls -lh ~/salus_data_temporal/

# Submit training job
sbatch slurm_train.sh

# Monitor
tail -f logs/train_*.out

# Check status
squeue -u $USER
```

**The job will**:
- Train for 100 epochs (~4 hours)
- Use 6 cores, 16GB RAM, 1 GPU
- Save model: `~/SalusTest/checkpoints/temporal_production/best_model.pt`
- **Automatically backup checkpoints**
- Email you when complete

### 7. Download Results (Optional)

```bash
# On LOCAL machine
export HPC_HOST=athene-login.hpc.fau.edu
export HPC_USER=your_username

cd /home/mpcr/Desktop/Salus\ Test/SalusTest
./sync_from_hpc.sh
```

Downloads to:
- Data: `~/salus_data_hpc/`
- Checkpoints: `~/SalusTest/checkpoints_hpc/`

---

## 📊 Resource Usage (All Within Your Limits!)

| Job | Cores | RAM | GPU | Time | Your Limit |
|-----|-------|-----|-----|------|------------|
| **Test** | 2 | 8GB | 1 | 30 min | ✅ 6/16GB |
| **Collection** | 6 | 16GB | 1 | 48-72h | ✅ 6/16GB |
| **Training** | 6 | 16GB | 1 | 4-8h | ✅ 6/16GB |

**All jobs fit perfectly!** We optimized for your constraints.

---

## 🎯 What Each Script Does

### SLURM Job Scripts

1. **`slurm_test_phase1.sh`** (30 min)
   - Validates Python/CUDA/GPU
   - Runs 7 component tests
   - Quick proof test
   - **Run first** to verify everything works

2. **`slurm_collect_data.sh`** (48-72 hours)
   - Collects 500 episodes
   - Uses 2 parallel environments (reduced for your limits)
   - Outputs ~10-15GB data
   - **Auto-backups to cloud** when done
   - Emails you when complete

3. **`slurm_train.sh`** (4-8 hours)
   - Trains temporal predictor
   - 100 epochs, batch size 64
   - Hard negative mining enabled
   - FP16 mixed precision
   - **Auto-backups checkpoints**
   - Emails you when complete

### Utility Scripts

4. **`sync_to_hpc.sh`** (10 sec)
   - Uploads code to HPC
   - Only 2MB (excludes venv, data)
   - Run from local machine

5. **`sync_from_hpc.sh`** (30-60 min)
   - Downloads data from HPC
   - Run from local machine
   - Optional (you have cloud backup)

6. **`setup_rclone_hpc.sh`** (5 min)
   - Installs rclone on HPC
   - Run once on HPC
   - Needed for cloud backup

7. **`backup_from_hpc.sh`** (30 min)
   - Backs up data to cloud
   - Auto-runs after collection/training
   - Can run manually if needed

---

## 📧 Email Notifications

**IMPORTANT**: Edit your email in SLURM scripts!

```bash
# On HPC, edit each slurm_*.sh file
nano slurm_collect_data.sh

# Change this line:
#SBATCH --mail-user=your_email@example.com  # ← PUT YOUR EMAIL HERE
```

You'll get emails for:
- ✅ Job completed successfully
- ❌ Job failed
- ⏰ Job timed out

---

## 🔍 Monitoring

### Check Job Status

```bash
# On HPC
squeue -u $USER

# Output:
# JOBID  PARTITION  NAME           USER  ST  TIME  NODES  NODELIST
# 12345  gpu        salus_collect  you   R   2:30  1      node01
```

### View Logs

```bash
# Real-time monitoring
tail -f logs/collect_*.out

# Check completed job
cat logs/test_*.out
less logs/train_*.out
```

### Check Disk Usage

```bash
# On HPC
du -sh ~/salus_data_temporal/    # Data directory
du -sh ~/SalusTest/checkpoints/  # Checkpoints
quota -s                          # Your quota
```

### Cancel Job

```bash
# If needed
scancel JOBID
```

---

## ⚠️ Important Notes

### 1. BACKUP IS AUTOMATIC! ✅

Both collection and training jobs **automatically backup** when complete:
- Data → Google Drive (or your configured remote)
- Checkpoints → Google Drive

**You don't need to do anything!** Just make sure rclone is configured.

### 2. Resource Limits Respected

All jobs use:
- ≤ 6 CPU cores (your limit)
- ≤ 16GB RAM (your limit)
- 1 GPU

We **reduced** `num_envs` from 4 → 2 to fit your limits.

### 3. Data is Safe

You have **TRIPLE backup**:
1. HPC storage (temporary)
2. Cloud backup (automatic)
3. Local copy (optional)

### 4. Jobs Email You

You'll know when:
- Collection finishes (48 hours)
- Training finishes (4 hours)
- Any job fails

No need to constantly check!

---

## 🐛 Troubleshooting

### Sync fails

```bash
# Test SSH
ssh your_username@athene-login.hpc.fau.edu

# If that fails, check SSH keys
ssh-keygen
ssh-copy-id your_username@athene-login.hpc.fau.edu
```

### "Module not found"

```bash
# On HPC, install dependencies
pip install torch torchvision numpy zarr tqdm tensorboard scikit-learn matplotlib

# Or load modules
module load python/3.10
module load cuda/11.8
```

### Job pending forever

```bash
# Check why
squeue -j JOBID --start

# Common reasons:
# - QOSMaxCpuPerUserLimit: You have other jobs
# - Resources: Cluster busy
# - Priority: Low priority (just wait)
```

### Out of memory

```bash
# Reduce batch size in slurm_train.sh
BATCH_SIZE=32  # Instead of 64

# Or reduce RAM request
#SBATCH --mem=12G  # Instead of 16G
```

### rclone not working

```bash
# On HPC
bash setup_rclone_hpc.sh
~/bin/rclone config
~/bin/rclone lsd backup:
```

---

## 📋 Complete Timeline

| Day | Action | Duration | Active/Waiting |
|-----|--------|----------|----------------|
| **Day 1** | Sync code | 10 sec | Active |
| | Setup HPC | 10 min | Active |
| | Run tests | 30 min | SLURM |
| | Start collection | Submit | Active |
| **Day 2-3** | Data collection | 48h | SLURM (wait) |
| | Auto-backup | 30 min | Auto |
| | Start training | Submit | Active |
| | Training | 4h | SLURM (wait) |
| | Auto-backup | 5 min | Auto |
| **Optional** | Download to local | 30 min | Active |

**Total active time**: ~1 hour
**Total wait time**: ~52 hours (automated)

---

## ✅ Pre-Flight Checklist

Before launching:

- [ ] HPC credentials configured (`HPC_HOST`, `HPC_USER`)
- [ ] SSH access tested
- [ ] Email address added to SLURM scripts
- [ ] Understand you have 6 core / 16GB limits (all jobs fit!)
- [ ] Understand jobs will take 2-3 days (mostly waiting)
- [ ] Know data will be automatically backed up to cloud

---

## 🚀 LAUNCH!

**Everything is ready!** Just run:

```bash
# 1. Set credentials
export HPC_HOST=athene-login.hpc.fau.edu
export HPC_USER=your_username

# 2. Sync (10 seconds)
cd /home/mpcr/Desktop/Salus\ Test/SalusTest
./sync_to_hpc.sh

# 3. SSH and start
ssh your_username@athene-login.hpc.fau.edu
cd ~/SalusTest
sbatch slurm_test_phase1.sh
```

**That's it!** The rest is automated. 🎉

---

## 📞 Need Help?

Check these docs:
- **HPC_SYNC_DETAILS.md** - Exact sync details
- **DATA_BACKUP_STRATEGY.md** - Backup guide
- **HPC_QUICKSTART.md** - Complete walkthrough
- **HPC_READY.md** - Deployment status

---

**Last Updated**: January 6, 2026
**Status**: ✅ ALL SYSTEMS GO
**Next**: Run `./sync_to_hpc.sh`
