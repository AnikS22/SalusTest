# SALUS HPC Deployment Guide

**Quick reference for deploying SALUS on A100 HPC**

---

## 📋 Prerequisites

- HPC account with A100 GPU access
- SSH access to HPC
- Google Cloud account (for backup)
- Isaac Lab installed on HPC (or instructions to install)

---

## 🚀 Quick Start (Step-by-Step)

### **Step 1: Transfer Files to HPC**

From your local machine:

```bash
# Package SALUS codebase
cd "/home/mpcr/Desktop/Salus Test/SalusTest"
tar -czf salus_deployment.tar.gz \
    salus/ \
    scripts/ \
    configs/ \
    hpc_setup.sh \
    setup_rclone.sh \
    test_salus.sh \
    deploy_a100.sh \
    A100_SCALING_GUIDE.md \
    DATA_MANAGEMENT_GUIDE.md

# Transfer to HPC (replace with your details)
scp salus_deployment.tar.gz username@hpc_address:~/

# SSH into HPC
ssh username@hpc_address
```

### **Step 2: Extract and Setup**

On HPC:

```bash
# Extract files
cd ~
tar -xzf salus_deployment.tar.gz -C salus_a100/
cd salus_a100

# Make scripts executable
chmod +x hpc_setup.sh setup_rclone.sh test_salus.sh deploy_a100.sh

# Run initial setup (5-10 minutes)
bash hpc_setup.sh
```

**What this does:**
- ✓ Verifies A100 GPU
- ✓ Sets up conda environment
- ✓ Installs dependencies (zarr, torch, lerobot)
- ✓ Downloads SmolVLA model (~2GB)
- ✓ Creates directory structure

### **Step 3: Configure Google Cloud Backup**

```bash
# Setup Rclone for automated backups
bash setup_rclone.sh
```

**Follow the interactive prompts:**
1. Configure rclone with Google Cloud
2. Create bucket (e.g., `salus-a100-data`)
3. Test connection
4. Set up automated backups (optional: every 6 hours)

### **Step 4: Test Pipeline (1 Episode)**

```bash
# Test with 1 episode (~13 minutes)
bash test_salus.sh
```

**This verifies:**
- ✓ Isaac Lab works
- ✓ VLA ensemble loads
- ✓ Data collection works
- ✓ All sensors recording

### **Step 5: Deploy Full Collection**

```bash
# Start 500-episode collection (~50 hours)
nohup bash deploy_a100.sh collect 500 8 > deployment.log 2>&1 &

# Save the process ID
echo $! > collection.pid
```

**Monitor progress:**
```bash
# Watch real-time progress
tail -f a100_logs/collection_500eps_*.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check backup status (if enabled)
tail -f backup_logs/backup_*.log
```

---

## 📊 Expected Timeline

| Phase | Duration | What Happens |
|-------|----------|-------------|
| Setup | 10-15 min | Install dependencies, download models |
| Rclone Config | 5 min | Configure Google Cloud backup |
| Pipeline Test | 13 min | Verify 1 episode works |
| **Data Collection** | **~50 hours** | **Collect 500 episodes with 8 parallel envs** |
| Backup | ~10 min | Sync data to Google Cloud |
| Training | 15 min | Train large model with FP16 |
| Evaluation | 5 min | Compute per-horizon metrics |

**Total: ~51 hours (mostly unattended)**

---

## 🔧 Common Operations

### Check Collection Progress

```bash
# View latest log
tail -n 50 a100_logs/collection_500eps_*.log

# Count episodes collected
ls -d a100_data/training_500eps/*/data.zarr | wc -l

# Check disk usage
du -sh a100_data/
```

### Manual Backup

```bash
# Run backup script manually
~/salus_a100/auto_backup.sh

# Check what's in cloud
rclone ls salus_gcloud:salus-a100-data/

# Verify backup size
rclone size salus_gcloud:salus-a100-data/
```

### Resume Interrupted Collection

If collection stops (network issue, timeout, etc.):

```bash
# Check last PID
cat collection.pid

# Verify process is dead
ps aux | grep collect_data_parallel_a100

# Resume from checkpoint
nohup bash deploy_a100.sh collect 500 8 > deployment_resume.log 2>&1 &
echo $! > collection.pid
```

**The script automatically resumes from last checkpoint!**

### Kill Collection

```bash
# Get process ID
cat collection.pid

# Kill process
kill $(cat collection.pid)

# Or kill all Python processes (careful!)
pkill -f collect_data_parallel_a100
```

---

## 📈 After Collection Completes

### Train Model

```bash
# Automatic training after collection
bash deploy_a100.sh train

# Or train manually
python scripts/train_failure_predictor_a100.py \
    --data_path a100_data/training_500eps/[timestamp]/data.zarr \
    --save_dir a100_checkpoints \
    --batch_size 1024 \
    --num_epochs 100 \
    --model_size large \
    --use_amp \
    --num_workers 8
```

### Evaluate Results

```bash
# Per-horizon evaluation
python scripts/evaluate_temporal_forecasting.py \
    --model_path a100_checkpoints/best_predictor_a100.pt \
    --data_path a100_data/test_150eps/[timestamp]/data.zarr \
    --save_dir a100_results

# View results
cat a100_results/temporal_forecasting_results.json
```

### Download Results to Local Machine

```bash
# From your local machine
scp -r username@hpc_address:~/salus_a100/a100_results/ ./
scp -r username@hpc_address:~/salus_a100/a100_checkpoints/ ./
```

---

## 🚨 Troubleshooting

### GPU Out of Memory

**Problem:** CUDA OOM error during collection

**Solution:** Reduce parallel environments
```bash
# Use 4 instead of 8 environments
bash deploy_a100.sh collect 500 4
```

### Isaac Lab Import Error

**Problem:** `ModuleNotFoundError: No module named 'isaaclab'`

**Solution:** Install Isaac Lab
```bash
# Clone Isaac Lab
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Follow installation instructions
./isaaclab.sh --install
```

### SmolVLA Download Fails

**Problem:** Model download timeout or fails

**Solution:** Download manually
```bash
conda activate isaaclab
pip install huggingface_hub
mkdir -p ~/models/smolvla
cd ~/models/smolvla
huggingface-cli download HuggingFaceTB/smolvla-base --local-dir smolvla_base
```

### Rclone Authentication Issues

**Problem:** Can't authenticate with Google Cloud

**Solution:** Use service account instead of OAuth
```bash
# 1. Create service account in GCP console
# 2. Download JSON key file
# 3. Transfer to HPC
scp service-account-key.json username@hpc_address:~/

# 4. Reconfigure rclone with service account
rclone config
# Choose service_account_file and provide path to JSON
```

### Collection Seems Stuck

**Problem:** No progress for >30 minutes

**Check:**
```bash
# GPU usage (should be 30-40GB)
nvidia-smi

# Process is running
ps aux | grep collect_data_parallel_a100

# Check log for errors
tail -n 100 a100_logs/collection_*.log
```

If truly stuck:
```bash
# Kill and restart
kill $(cat collection.pid)
bash deploy_a100.sh collect 500 8
```

---

## 💰 Cost Tracking

### Compute Costs
- A100 time: ~50 hours @ $X/hour (check your HPC pricing)
- Storage: ~1.5GB on HPC

### Google Cloud Costs
- Storage (Standard): ~$0.02/month for 1.5GB
- Egress: Usually free for first download
- **Total estimate: <$1/month** for data storage

### Cost Optimization
```bash
# After publication, move to Coldline storage
gsutil rewrite -s COLDLINE gs://salus-a100-data/experiments/old_run_*

# Or via rclone
rclone backend set-tier salus_gcloud:salus-a100-data/experiments/old_run_1 COLDLINE
```

---

## 📞 Quick Reference Commands

```bash
# Setup
bash hpc_setup.sh              # Initial setup
bash setup_rclone.sh           # Configure backups
bash test_salus.sh             # Test with 1 episode

# Deployment
bash deploy_a100.sh collect 500 8    # Collect data
bash deploy_a100.sh train            # Train model
bash deploy_a100.sh both 500 8       # Collect + train

# Monitoring
tail -f a100_logs/collection_*.log   # Watch progress
watch -n 1 nvidia-smi                # GPU usage
~/salus_a100/auto_backup.sh          # Manual backup

# Results
cat a100_checkpoints/training_results_a100.json
cat a100_results/temporal_forecasting_results.json
```

---

## ✅ Success Criteria

Before considering deployment complete:

- [ ] 500+ episodes collected
- [ ] Success rate 30-40% in data
- [ ] All 4 failure types present (>5% each)
- [ ] Model trains without OOM
- [ ] Test F1 > 0.60
- [ ] Test Precision > 0.50
- [ ] Test Recall > 0.60
- [ ] Per-horizon metrics computed
- [ ] Data backed up to Google Cloud
- [ ] Results downloaded to local machine

---

## 📚 Additional Resources

- **A100_SCALING_GUIDE.md**: Complete technical details
- **DATA_MANAGEMENT_GUIDE.md**: Rclone workflows and strategies
- **PROOF_OF_CONCEPT_RESULTS.md**: Baseline metrics (F1=0.327)

---

**Need Help?** Check the logs:
- Setup issues: `cat hpc_setup.log`
- Collection issues: `tail -100 a100_logs/collection_*.log`
- Training issues: `tail -100 a100_logs/training_*.log`
- Backup issues: `tail -100 backup_logs/backup_*.log`
