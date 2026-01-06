# SALUS Deployment on Athene HPC (FAU)

**Specific instructions for asahai2024@athene-login.hpc.fau.edu**

---

## Quick Connection Commands

```bash
# Transfer package (from local machine)
cd "/home/mpcr/Desktop/Salus Test/SalusTest"
scp salus_a100_deployment_20260105_200242.tar.gz asahai2024@athene-login.hpc.fau.edu:~/

# SSH into Athene
ssh asahai2024@athene-login.hpc.fau.edu
```

---

## Athene HPC Specifics

### Job Scheduler
Athene likely uses **SLURM** or **PBS** for job scheduling. GPU nodes are typically not available on login nodes.

### Check Available Resources

```bash
# For SLURM
sinfo -p gpu           # List GPU partitions
squeue -u asahai2024   # Your current jobs

# For PBS
pbsnodes -a            # List all nodes
qstat -u asahai2024    # Your current jobs
```

### Request Interactive GPU Session

**Option 1: Interactive Session (for testing)**
```bash
# SLURM
srun --partition=gpu-a100 --gpus=1 --ntasks=8 --mem=64G \
     --time=02:00:00 --pty bash

# PBS
qsub -I -l select=1:ngpus=1:mem=64gb -l walltime=02:00:00
```

**Option 2: Batch Job (for 50-hour collection)**
Create a job script (recommended for long runs):

```bash
cat > ~/submit_salus.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=salus_collection
#SBATCH --partition=gpu-a100
#SBATCH --gpus=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --time=60:00:00
#SBATCH --output=salus_%j.out
#SBATCH --error=salus_%j.err

# Load modules (if required by Athene)
module load cuda/11.8
module load python/3.10

# Navigate to SALUS directory
cd ~/salus_a100_deployment_20260105_200242

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab

# Run deployment
bash deploy_a100.sh collect 500 8
EOF

chmod +x ~/submit_salus.sh

# Submit job
sbatch ~/submit_salus.sh
```

---

## Step-by-Step Deployment

### 1. Initial Setup (Interactive Session Recommended)

```bash
# SSH into Athene
ssh asahai2024@athene-login.hpc.fau.edu

# Request interactive GPU session for setup
srun --partition=gpu-a100 --gpus=1 --time=02:00:00 --pty bash

# Extract deployment package
cd ~
tar -xzf salus_a100_deployment_20260105_200242.tar.gz
cd salus_a100_deployment_20260105_200242

# Run setup (installs conda, downloads models, ~15 min)
bash hpc_setup.sh
```

### 2. Configure Google Cloud Backup

```bash
# Still in interactive session
bash setup_rclone.sh
```

Follow prompts:
1. Configure `salus_gcloud` remote
2. Create bucket (e.g., `salus-athene-a100`)
3. Test connection
4. Setup cron backup (optional)

### 3. Test Pipeline (1 Episode)

```bash
# Test with 1 episode (~13 minutes)
bash test_salus.sh
```

If successful: ✓ Pipeline Test PASSED!

### 4. Submit Long-Running Collection Job

**Exit interactive session** (Ctrl+D) and return to login node.

**Option A: Submit Batch Job (Recommended)**
```bash
# Edit the submit script with correct paths
cd ~/salus_a100_deployment_20260105_200242

# Submit 50-hour collection job
sbatch submit_salus.sh

# Check job status
squeue -u asahai2024

# View output
tail -f salus_*.out
```

**Option B: Long Interactive Session**
```bash
# Request 60-hour interactive session
srun --partition=gpu-a100 --gpus=1 --ntasks=8 \
     --time=60:00:00 --pty bash

# Run collection
cd ~/salus_a100_deployment_20260105_200242
bash deploy_a100.sh collect 500 8
```

---

## Monitoring from Login Node

While job is running:

```bash
# Check job status
squeue -u asahai2024

# View output logs
tail -f ~/salus_*.out
tail -f ~/salus_*.err

# Check collected episodes (if NFS mounted)
find ~/salus_a100/a100_data -name "data.zarr" -type d | wc -l

# Monitor backups
tail -f ~/salus_a100/backup_logs/backup_*.log
```

---

## Athene-Specific Considerations

### Storage Quotas
Check your storage quota:
```bash
quota -s                    # Check quota
du -sh ~/salus_a100/        # Check SALUS usage
```

**Expected usage:**
- Initial: ~2 GB (models)
- During collection: ~5-10 GB
- Final: ~1.5 GB (500 episodes compressed)

### Module System
If Athene requires module loading:
```bash
# Check available modules
module avail

# Typical requirements
module load cuda/11.8 or cuda/12.1
module load python/3.10
module load conda
```

Add these to your job script if needed.

### Network/Internet Access
Some HPC compute nodes may not have internet access. If SmolVLA download fails:

**On login node (has internet):**
```bash
# Download model on login node
conda activate isaaclab
pip install huggingface_hub
mkdir -p ~/models/smolvla
cd ~/models/smolvla
huggingface-cli download HuggingFaceTB/smolvla-base --local-dir smolvla_base
```

**Then in compute job:** Model will be loaded from `~/models/smolvla/smolvla_base`

---

## After Collection Completes

### Download Results to Local Machine

From your local machine:

```bash
# Download trained model
scp asahai2024@athene-login.hpc.fau.edu:~/salus_a100/a100_checkpoints/best_predictor_a100.pt ./

# Download results
scp -r asahai2024@athene-login.hpc.fau.edu:~/salus_a100/a100_results/ ./

# Download training logs
scp asahai2024@athene-login.hpc.fau.edu:~/salus_a100/a100_checkpoints/training_results_a100.json ./
```

### Cleanup (After Backup Verified)

```bash
# On Athene - only after confirming Google Cloud backup!
cd ~/salus_a100

# Verify backup first
rclone ls salus_gcloud:salus-athene-a100/

# Then remove local data (keep checkpoints)
rm -rf a100_data/

# Keep models and results
ls -lh a100_checkpoints/
ls -lh a100_results/
```

---

## Troubleshooting Athene-Specific Issues

### Issue: Cannot access GPU on login node
**Solution:** Don't run on login node - request interactive session or submit job

### Issue: Job killed for exceeding walltime
**Solution:** Request longer walltime in job script (60 hours minimum)

### Issue: Out of storage quota
**Solution:**
- Enable Rclone backups to Google Cloud
- Delete old test runs
- Use scratch space if available

### Issue: Module load errors
**Solution:** Check required modules with HPC support:
```bash
# Contact support or check documentation
module spider cuda
module spider isaac
```

---

## Quick Reference

```bash
# Connection
ssh asahai2024@athene-login.hpc.fau.edu

# Request GPU (SLURM)
srun --partition=gpu-a100 --gpus=1 --time=02:00:00 --pty bash

# Submit job (SLURM)
sbatch submit_salus.sh

# Check status
squeue -u asahai2024

# Monitor
tail -f salus_*.out
tail -f ~/salus_a100/a100_logs/collection_*.log

# Count episodes
find ~/salus_a100/a100_data -name "data.zarr" -type d | wc -l
```

---

## Support Contacts

**Athene HPC Support:**
- Documentation: Check Athene HPC user guide
- Email: (check your HPC portal for support email)
- Ticket system: (check your HPC portal)

**Common questions to ask HPC support:**
1. What GPU partitions are available? (`gpu-a100`, `gpu`, etc.)
2. What is the maximum walltime for GPU jobs?
3. Are compute nodes allowed internet access?
4. What CUDA modules are available?
5. Is there scratch space for large temporary files?

---

## Timeline for Athene Deployment

| Time | Action | Location |
|------|--------|----------|
| Now | Transfer package | Local → Athene |
| +2 min | SSH and extract | Login node |
| +5 min | Request GPU session | GPU node |
| +20 min | Setup + test | GPU node |
| +25 min | Submit batch job | Login node |
| +50 hours | Collection completes | GPU node |
| +50h 15m | Train model | GPU node |
| +50h 20m | Download results | Athene → Local |

**Total: ~50 hours (mostly unattended batch job)**

---

Good luck with your deployment on Athene! 🚀
