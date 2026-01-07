# SALUS on Athene - Quick Start

**Your username**: asahai2024
**Cluster**: athene-login.hpc.fau.edu

---

## 🚀 OPTION 1: Quick Manual Sync (Recommended - 2 minutes)

Run this in your terminal (you'll need to enter your Athene password):

```bash
cd /home/mpcr/Desktop/Salus\ Test/SalusTest

# This will sync all code (~19MB)
./SYNC_TO_ATHENE.sh
```

**You'll be prompted for password 2 times**:
1. First time: Creating directory
2. Second time: Syncing files

**Uploads**: 126 files (~19MB) in ~30 seconds

---

## 🚀 OPTION 2: Manual Commands (If script doesn't work)

```bash
# 1. Test connection (enter password when prompted)
ssh asahai2024@athene-login.hpc.fau.edu

# You should see Athene login prompt
# Type 'exit' to close connection

# 2. Create directory on Athene
ssh asahai2024@athene-login.hpc.fau.edu "mkdir -p ~/SalusTest"

# 3. Sync code (enter password when prompted)
cd /home/mpcr/Desktop/Salus\ Test/SalusTest

rsync -avz --progress \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='venv*/' \
    --exclude='data/' \
    --exclude='checkpoints/' \
    --exclude='logs/' \
    --exclude='*.zarr/' \
    --exclude='paper_data/' \
    --exclude='models/' \
    ./ asahai2024@athene-login.hpc.fau.edu:~/SalusTest/
```

---

## ✅ After Sync - Run on Athene

```bash
# 1. SSH to Athene
ssh asahai2024@athene-login.hpc.fau.edu

# 2. Navigate to project
cd ~/SalusTest

# 3. Check files synced
ls -lh
# Should see: salus/ scripts/ docs/ *.sh *.md

# 4. Make scripts executable
chmod +x slurm_*.sh *.sh

# 5. Create logs directory
mkdir -p logs

# 6. Run Phase 1 tests
python scripts/test_hpc_phase1.py
```

**Expected output**: All 4 tests pass ✅

---

## 📦 What Gets Synced

```
UPLOADED (~19MB):
├── salus/          408KB   ← All Python code
├── scripts/        228KB   ← All scripts
├── docs/           832KB   ← Documentation
├── slurm_*.sh      20KB    ← SLURM job scripts
├── *.sh            50KB    ← Backup/sync scripts
└── *.md            ~18MB   ← Guides

NOT UPLOADED (Automatic):
├── venv_salus/     5.9GB   ← Too big
├── data/           10GB    ← Too big
├── .git/           ← Not needed
└── __pycache__/    ← Cache
```

---

## 🎯 Quick Test Sequence

After sync, run these on Athene:

```bash
# Run all tests
cd ~/SalusTest
python scripts/test_hpc_phase1.py

# Should see:
# ✅ Python/CUDA check: PASS
# ✅ Imports: PASS
# ✅ Component tests (7/7): PASS
# ✅ Quick proof: PASS (97.5% discrimination)
```

---

## 🚀 Launch Jobs (After Tests Pass)

### Small Test First (Recommended - 2 hours)

```bash
# On Athene
cd ~/SalusTest

# Collect 50 episodes (quick test)
sbatch --export=NUM_EPISODES=50,NUM_ENVS=1 slurm_collect_data.sh

# Monitor
squeue -u asahai2024
tail -f logs/collect_*.out
```

### Full Production (After Small Test Works)

```bash
# On Athene
cd ~/SalusTest

# Collect 500 episodes (48+ hours)
sbatch slurm_collect_data.sh

# Train model (4 hours)
sbatch slurm_train.sh

# Check status
squeue -u asahai2024
```

---

## 📧 Email Notifications

**IMPORTANT**: Add your email to SLURM scripts!

```bash
# On Athene
cd ~/SalusTest
nano slurm_collect_data.sh

# Change this line:
#SBATCH --mail-user=your_email@example.com

# Same for other scripts:
nano slurm_train.sh
nano slurm_test_phase1.sh
```

---

## 🛡️ Setup Backup (Do After First Test)

```bash
# On Athene
cd ~/SalusTest

# Install rclone
bash setup_rclone_hpc.sh

# Configure Google Drive
~/bin/rclone config
# n) New remote
# name> backup
# Storage> drive
# [Follow authorization]

# Test
~/bin/rclone lsd backup:
```

---

## 🔍 Monitoring Commands

```bash
# Check your jobs
squeue -u asahai2024

# View logs
tail -f logs/collect_*.out

# Check disk space
du -sh ~/salus_data_temporal/
quota -s

# Cancel job if needed
scancel JOBID
```

---

## ⚡ Resource Limits (You Have)

- **CPUs**: 6 cores ✅
- **RAM**: 16GB ✅
- **GPU**: 1 GPU ✅
- **Time**: 72 hours max per job ✅

**All our jobs fit within your limits!**

---

## 📋 Complete Workflow Summary

```bash
# === ON LOCAL MACHINE ===
cd /home/mpcr/Desktop/Salus\ Test/SalusTest
./SYNC_TO_ATHENE.sh
# (Enter password when prompted)

# === ON ATHENE ===
ssh asahai2024@athene-login.hpc.fau.edu
cd ~/SalusTest

# Test
python scripts/test_hpc_phase1.py

# Small test (2 hours)
sbatch --export=NUM_EPISODES=50,NUM_ENVS=1 slurm_collect_data.sh

# Full production (after small test works)
sbatch slurm_collect_data.sh  # 48+ hours
sbatch slurm_train.sh          # 4 hours
```

---

## 🐛 Troubleshooting

### "Permission denied (publickey)"

You need to enter your password when syncing. The script will prompt you.

### "Module not found" on Athene

```bash
# Install dependencies
pip install torch torchvision numpy zarr tqdm tensorboard scikit-learn matplotlib

# Or use module system
module load python/3.10
module load cuda/11.8
```

### "Cannot find scripts"

```bash
# Make sure you're in the right directory
cd ~/SalusTest
ls -la
# Should see: salus/ scripts/ docs/
```

---

## ✅ Success Checklist

- [ ] Code synced to Athene (~19MB)
- [ ] SSH access working
- [ ] Phase 1 tests pass (4/4)
- [ ] Email added to SLURM scripts
- [ ] Small test submitted (50 episodes)
- [ ] Small test completes successfully
- [ ] Full collection submitted (500 episodes)
- [ ] Training submitted
- [ ] Backup configured

---

## 🎉 You're Ready!

Just run:

```bash
./SYNC_TO_ATHENE.sh
```

Then SSH to Athene and test! 🚀

---

**Last Updated**: January 6, 2026
**Username**: asahai2024
**Cluster**: athene-login.hpc.fau.edu
