# SALUS Data Management with Google Cloud + Rclone

**Managing A100 HPC data with Google Cloud Storage**

---

## 📊 Data Size Estimates

| Dataset | Episodes | Size | Compressed |
|---------|----------|------|------------|
| PoC (RTX 2080 Ti) | 50 | 67 MB | 67 MB |
| A100 Training | 500 | 670 MB | ~400 MB |
| A100 Validation | 150 | 200 MB | ~120 MB |
| A100 Test | 150 | 200 MB | ~120 MB |
| **Total** | **800** | **~1.1 GB** | **~650 MB** |

### With 1000 Episodes
| Dataset | Episodes | Size | Compressed |
|---------|----------|------|------------|
| Training | 1000 | 1.3 GB | ~800 MB |
| Validation | 200 | 270 MB | ~160 MB |
| Test | 200 | 270 MB | ~160 MB |
| **Total** | **1400** | **~1.85 GB** | **~1.1 GB** |

---

## 🚀 Quick Setup

### 1. Install Rclone on HPC

```bash
# On HPC A100 machine
curl https://rclone.org/install.sh | sudo bash

# Verify installation
rclone version
```

### 2. Configure Google Cloud Storage

```bash
# Interactive configuration
rclone config

# Follow these steps:
# n) New remote
# name> salus_gcloud
# Storage> google cloud storage (select number for gcs)
# project_number> [your-gcp-project-id]
# service_account_file> [leave blank for OAuth]
# object_acl> private
# bucket_acl> private
# location> us (or your preferred region)
# storage_class> STANDARD
# Edit advanced config? n
# Use auto config? y (opens browser for OAuth)

# Test connection
rclone lsd salus_gcloud:
```

### 3. Create Google Cloud Bucket

```bash
# Option 1: Via rclone
rclone mkdir salus_gcloud:salus-a100-data

# Option 2: Via gcloud CLI
gcloud storage buckets create gs://salus-a100-data \
    --location=us \
    --uniform-bucket-level-access

# Verify
rclone lsd salus_gcloud:
```

---

## 📁 Recommended Directory Structure

### On HPC (Local)
```
/scratch/username/salus/  (or /home/username/salus/)
├── proof_of_concept_rtx2080ti/       # PoC results (67 MB)
├── a100_data/
│   ├── training_500eps/              # 670 MB
│   ├── validation_150eps/            # 200 MB
│   └── test_150eps/                  # 200 MB
├── a100_checkpoints/                 # ~5 MB
├── a100_logs/                        # ~100 MB
└── a100_results/                     # ~10 MB
```

### On Google Cloud Storage
```
gs://salus-a100-data/
├── experiments/
│   ├── poc_rtx2080ti/                # Archived PoC
│   ├── a100_run1_500eps/             # First A100 run
│   ├── a100_run2_1000eps/            # Scaled run
│   └── a100_final_publication/       # Final dataset for paper
├── checkpoints/                      # All trained models
├── results/                          # Analysis and figures
└── logs/                             # Training/collection logs
```

---

## 🔄 Data Sync Workflows

### Workflow 1: Continuous Backup During Collection

**Use case**: Backup data as it's being collected on HPC

```bash
#!/bin/bash
# sync_continuous.sh - Run this while collection is ongoing

# Sync every hour
while true; do
    echo "[$(date)] Syncing to Google Cloud..."

    rclone sync \
        /scratch/username/salus/a100_data/ \
        salus_gcloud:salus-a100-data/experiments/a100_run1_500eps/data/ \
        --progress \
        --transfers 4 \
        --checkers 8 \
        --log-file rclone_sync.log

    echo "[$(date)] Sync complete. Waiting 1 hour..."
    sleep 3600  # 1 hour
done
```

**Run in background**:
```bash
chmod +x sync_continuous.sh
nohup ./sync_continuous.sh > sync_continuous.log 2>&1 &
```

### Workflow 2: Post-Collection Full Backup

**Use case**: After collection completes, do a full backup

```bash
#!/bin/bash
# backup_full.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="a100_run1_500eps_${TIMESTAMP}"

echo "Backing up complete experiment: $RUN_NAME"

# Backup data
rclone copy \
    /scratch/username/salus/a100_data/ \
    salus_gcloud:salus-a100-data/experiments/${RUN_NAME}/data/ \
    --progress \
    --transfers 8

# Backup checkpoints
rclone copy \
    /scratch/username/salus/a100_checkpoints/ \
    salus_gcloud:salus-a100-data/checkpoints/${RUN_NAME}/ \
    --progress

# Backup results
rclone copy \
    /scratch/username/salus/a100_results/ \
    salus_gcloud:salus-a100-data/results/${RUN_NAME}/ \
    --progress

# Backup logs
rclone copy \
    /scratch/username/salus/a100_logs/ \
    salus_gcloud:salus-a100-data/logs/${RUN_NAME}/ \
    --progress

echo "✅ Backup complete: $RUN_NAME"
```

### Workflow 3: Download for Local Analysis

**Use case**: Download results to local machine for paper writing

```bash
# On your local machine
rclone copy \
    salus_gcloud:salus-a100-data/results/a100_run1_500eps_20260106/ \
    ~/Desktop/salus_results/ \
    --progress

# Download specific checkpoints
rclone copy \
    salus_gcloud:salus-a100-data/checkpoints/a100_run1_500eps/best_predictor_a100.pt \
    ~/Desktop/salus_models/ \
    --progress
```

### Workflow 4: Incremental Sync (Efficient)

**Use case**: Only sync changes (faster, saves bandwidth)

```bash
# Sync only new/modified files
rclone sync \
    /scratch/username/salus/a100_data/ \
    salus_gcloud:salus-a100-data/experiments/current/ \
    --update \
    --use-server-modtime \
    --progress

# The --update flag only transfers files that are newer
```

---

## ⚡ Optimization Tips

### 1. Fast Transfer Settings

```bash
# For maximum speed (use with caution on shared HPC)
rclone sync source destination \
    --transfers 16 \          # Parallel transfers
    --checkers 32 \           # Parallel file checking
    --buffer-size 256M \      # Large buffer for big files
    --drive-chunk-size 256M \ # Large chunks
    --progress
```

### 2. Bandwidth Control

```bash
# Limit bandwidth (good for shared HPC)
rclone sync source destination \
    --bwlimit 10M \           # 10 MB/s limit
    --progress
```

### 3. Compression During Transfer

```bash
# Compress on the fly (saves bandwidth)
rclone copy source destination \
    --gcs-encoding "gzip" \
    --progress
```

### 4. Exclude Unnecessary Files

```bash
# Don't backup temporary files
rclone sync source destination \
    --exclude "*.log" \
    --exclude "*.tmp" \
    --exclude "__pycache__/**" \
    --progress
```

---

## 🔐 Security Best Practices

### 1. Use Service Account (Recommended for HPC)

```bash
# Create service account in GCP
gcloud iam service-accounts create salus-hpc \
    --display-name "SALUS HPC Data Transfer"

# Generate key
gcloud iam service-accounts keys create ~/salus-key.json \
    --iam-account salus-hpc@[PROJECT-ID].iam.gserviceaccount.com

# Grant permissions
gcloud storage buckets add-iam-policy-binding gs://salus-a100-data \
    --member="serviceAccount:salus-hpc@[PROJECT-ID].iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# Configure rclone with service account
rclone config
# service_account_file> ~/salus-key.json
```

### 2. Encrypt Sensitive Data

```bash
# Encrypt before upload (if data contains sensitive info)
rclone copy source destination \
    --crypt-remote salus_gcloud_encrypted: \
    --progress
```

### 3. Set Object Lifecycle Rules

```bash
# Automatically archive old data after 90 days
gcloud storage buckets update gs://salus-a100-data \
    --lifecycle-file lifecycle.json

# lifecycle.json:
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {"age": 90}
      }
    ]
  }
}
```

---

## 💰 Cost Optimization

### Google Cloud Storage Pricing (approximate)

| Storage Class | Price/GB/month | Best For |
|---------------|----------------|----------|
| Standard | $0.020 | Active data, frequent access |
| Nearline | $0.010 | <1 access/month |
| Coldline | $0.004 | <1 access/quarter |
| Archive | $0.0012 | Long-term backup |

**Estimated Costs for SALUS**:
- 1.1 GB dataset in Standard: **~$0.02/month**
- 1.1 GB dataset in Coldline (after experiment): **~$0.005/month**

**Recommendation**: Use Standard during active development, move to Coldline after publication.

### Cost-Saving Strategy

```bash
# Move old experiments to cheaper storage
gsutil -m rewrite -s COLDLINE gs://salus-a100-data/experiments/old_run_*

# Or via rclone
rclone backend set-tier salus_gcloud:salus-a100-data/experiments/old_run_1 COLDLINE
```

---

## 🛠️ Automated Backup Script

Save this as `/scratch/username/salus/auto_backup.sh`:

```bash
#!/bin/bash
# Automated SALUS Data Backup to Google Cloud

set -e

# Configuration
LOCAL_BASE="/scratch/username/salus"
REMOTE_BASE="salus_gcloud:salus-a100-data"
RUN_NAME="a100_run1_500eps"
LOG_DIR="$LOCAL_BASE/backup_logs"

# Create log directory
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/backup_${TIMESTAMP}.log"

echo "==================================" | tee -a "$LOG_FILE"
echo "SALUS Backup: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "==================================" | tee -a "$LOG_FILE"

# Function to sync directory
sync_dir() {
    local src=$1
    local dest=$2
    local desc=$3

    echo "" | tee -a "$LOG_FILE"
    echo "Syncing $desc..." | tee -a "$LOG_FILE"

    if [ -d "$src" ]; then
        rclone sync "$src" "$dest" \
            --transfers 8 \
            --checkers 16 \
            --update \
            --progress \
            --log-file "$LOG_FILE" \
            --log-level INFO

        echo "✅ $desc synced" | tee -a "$LOG_FILE"
    else
        echo "⚠️  $desc not found: $src" | tee -a "$LOG_FILE"
    fi
}

# Sync all directories
sync_dir \
    "$LOCAL_BASE/a100_data" \
    "$REMOTE_BASE/experiments/${RUN_NAME}/data" \
    "Data"

sync_dir \
    "$LOCAL_BASE/a100_checkpoints" \
    "$REMOTE_BASE/checkpoints/${RUN_NAME}" \
    "Checkpoints"

sync_dir \
    "$LOCAL_BASE/a100_results" \
    "$REMOTE_BASE/results/${RUN_NAME}" \
    "Results"

sync_dir \
    "$LOCAL_BASE/a100_logs" \
    "$REMOTE_BASE/logs/${RUN_NAME}" \
    "Logs"

echo "" | tee -a "$LOG_FILE"
echo "==================================" | tee -a "$LOG_FILE"
echo "✅ Backup complete: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "==================================" | tee -a "$LOG_FILE"

# Show summary
echo "" | tee -a "$LOG_FILE"
echo "Summary:" | tee -a "$LOG_FILE"
rclone size "$REMOTE_BASE/experiments/${RUN_NAME}" 2>&1 | tee -a "$LOG_FILE"
```

**Usage**:
```bash
chmod +x auto_backup.sh

# Manual backup
./auto_backup.sh

# Scheduled backup (every 6 hours)
crontab -e
# Add line:
0 */6 * * * /scratch/username/salus/auto_backup.sh
```

---

## 📊 Monitoring and Verification

### Check Sync Status

```bash
# Compare local vs cloud
rclone check \
    /scratch/username/salus/a100_data/ \
    salus_gcloud:salus-a100-data/experiments/a100_run1_500eps/data/ \
    --combined rclone_check.txt

# Show differences
cat rclone_check.txt
```

### Monitor Storage Usage

```bash
# Local storage
du -sh /scratch/username/salus/*

# Cloud storage
rclone size salus_gcloud:salus-a100-data/
rclone ls salus_gcloud:salus-a100-data/ | head -20
```

### Verify Data Integrity

```bash
# Generate checksums locally
find /scratch/username/salus/a100_data -type f -exec md5sum {} \; > local_checksums.txt

# Compare with cloud
rclone hashsum MD5 salus_gcloud:salus-a100-data/experiments/a100_run1_500eps/data/ > cloud_checksums.txt

# Diff
diff <(sort local_checksums.txt) <(sort cloud_checksums.txt)
```

---

## 🚨 Disaster Recovery

### Scenario 1: HPC Data Loss

```bash
# Restore everything from cloud
rclone copy \
    salus_gcloud:salus-a100-data/experiments/a100_run1_500eps/ \
    /scratch/username/salus_restored/ \
    --progress

# Verify restoration
rclone check \
    /scratch/username/salus_restored/ \
    salus_gcloud:salus-a100-data/experiments/a100_run1_500eps/
```

### Scenario 2: Accidental Deletion

```bash
# Enable versioning (keeps old versions)
gcloud storage buckets update gs://salus-a100-data \
    --versioning

# List file versions
gsutil ls -a gs://salus-a100-data/experiments/*/data.zarr

# Restore specific version
gsutil cp gs://salus-a100-data/path/to/file#version /local/path
```

---

## ✅ Best Practices Checklist

Before starting A100 experiments:
- [ ] Rclone installed and configured on HPC
- [ ] Google Cloud bucket created
- [ ] Service account configured (if using)
- [ ] Auto-backup script set up
- [ ] Test sync with small file (verify it works)
- [ ] Scheduled backups configured (cron)
- [ ] Monitoring script in place

During experiments:
- [ ] Run sync every 6 hours (or continuously)
- [ ] Monitor backup logs for errors
- [ ] Verify critical checkpoints backed up immediately
- [ ] Check cloud storage usage regularly

After experiments:
- [ ] Final full backup to cloud
- [ ] Verify data integrity (checksums)
- [ ] Document experiment in cloud folder (README)
- [ ] Move to Coldline storage if archiving
- [ ] Clean up local HPC storage (after verification)

---

## 📞 Quick Reference Commands

```bash
# Basic sync
rclone sync /local/path salus_gcloud:remote/path --progress

# Copy (doesn't delete from destination)
rclone copy /local/path salus_gcloud:remote/path --progress

# List remote files
rclone ls salus_gcloud:salus-a100-data/

# Check differences
rclone check /local salus_gcloud:remote

# Size of remote directory
rclone size salus_gcloud:salus-a100-data/

# Mount cloud as local directory
rclone mount salus_gcloud:salus-a100-data /mnt/gcloud &

# Unmount
fusermount -u /mnt/gcloud
```

---

**Recommendation**: Start with the automated backup script running every 6 hours. This ensures you never lose more than 6 hours of work if something fails on the HPC.

**Total Setup Time**: ~30 minutes
**Ongoing Maintenance**: Minimal (automated backups handle everything)
**Cost**: ~$0.02-0.05/month for 1-2 GB of data
