# SALUS Data Backup Strategy with Rclone

## ✅ YES - Use Rclone for Data Backup!

**Rclone** is perfect for backing up SALUS data because:
- ✅ Works great with Zarr (cloud-native format)
- ✅ Syncs to S3, GCS, Azure, Dropbox, Google Drive
- ✅ Handles large datasets (50-100GB+)
- ✅ Incremental syncs (only uploads changes)
- ✅ Resume interrupted transfers
- ✅ Compression-friendly (Zarr is already compressed)

---

## Why Rclone + Zarr is Perfect

### Zarr is Cloud-Native
Zarr stores data as many small files (chunks), which is ideal for object storage:
```
data.zarr/
├── images/0.0.0.0  ← Small chunk file
├── images/0.0.0.1  ← Small chunk file
├── images/0.0.1.0  ← Small chunk file
...
```

**Benefits**:
- Rclone can sync individual chunks (efficient!)
- Failed uploads only retry failed chunks (not entire dataset)
- Parallel uploads of multiple chunks
- Works directly with S3/GCS (no conversion needed)

---

## Installation

```bash
# Install Rclone
curl https://rclone.org/install.sh | sudo bash

# Or on Ubuntu/Debian:
sudo apt install rclone

# Verify installation
rclone version
```

---

## Setup: Configure Cloud Storage

### Option 1: AWS S3 (Recommended for Large Datasets)

```bash
# Configure S3 remote
rclone config

# Follow prompts:
# n) New remote
# name: s3-salus-backup
# Storage: s3
# provider: AWS
# access_key_id: [your AWS access key]
# secret_access_key: [your AWS secret key]
# region: us-east-1 (or your preferred region)
# endpoint: (leave blank for AWS)
```

### Option 2: Google Cloud Storage (GCS)

```bash
rclone config
# name: gcs-salus-backup
# Storage: google cloud storage
# service_account_file: [path to service account JSON]
```

### Option 3: Backblaze B2 (Cheap Alternative)

```bash
rclone config
# name: b2-salus-backup
# Storage: b2
# account: [your B2 account ID]
# key: [your B2 application key]
```

### Option 4: Google Drive (Personal/Small Scale)

```bash
rclone config
# name: gdrive-salus-backup
# Storage: drive
# client_id: (leave blank, use defaults)
# client_secret: (leave blank, use defaults)
# scope: drive
# Follow OAuth flow in browser
```

---

## Backup Commands

### Basic Backup (One-time Sync)

```bash
# Backup data directory to S3
rclone sync /home/mpcr/Desktop/Salus\ Test/SalusTest/data s3-salus-backup:salus-data-backup/

# What this does:
# - sync: Makes destination match source (deletes files in destination if removed from source)
# - Use 'copy' instead of 'sync' if you don't want deletions
```

### Recommended: Incremental Backup Script

Create `scripts/backup_data.sh`:

```bash
#!/bin/bash
# SALUS Data Backup Script

# Configuration
LOCAL_DATA_DIR="/home/mpcr/Desktop/Salus Test/SalusTest/data"
REMOTE_NAME="s3-salus-backup"  # or gcs-salus-backup, b2-salus-backup, etc.
REMOTE_PATH="salus-data-backup"
LOG_FILE="/home/mpcr/Desktop/Salus Test/SalusTest/logs/backup.log"

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

# Backup with logging
echo "Starting backup at $(date)" >> "$LOG_FILE"
rclone sync "$LOCAL_DATA_DIR" "$REMOTE_NAME:$REMOTE_PATH/" \
    --progress \
    --transfers 16 \
    --checkers 32 \
    --log-file "$LOG_FILE" \
    --log-level INFO \
    --stats 30s \
    --retries 3 \
    --low-level-retries 10

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Backup completed successfully at $(date)" >> "$LOG_FILE"
else
    echo "❌ Backup failed at $(date)" >> "$LOG_FILE"
    exit 1
fi
```

Make it executable:
```bash
chmod +x scripts/backup_data.sh
```

### Automated Daily Backup (Cron)

```bash
# Edit crontab
crontab -e

# Add line for daily backup at 2 AM:
0 2 * * * /home/mpcr/Desktop/Salus\ Test/SalusTest/scripts/backup_data.sh
```

---

## Advanced: Backup Only Recent Data

### Backup Only New Episodes

```bash
# Only sync data.zarr directories modified in last 24 hours
rclone sync /home/mpcr/Desktop/Salus\ Test/SalusTest/data \
    s3-salus-backup:salus-data-backup/ \
    --max-age 24h \
    --include "data.zarr/**"
```

### Backup Specific Collection Runs

```bash
# Backup only overnight collection runs
rclone sync /home/mpcr/Desktop/Salus\ Test/SalusTest/data/mvp_episodes_overnight \
    s3-salus-backup:salus-data-backup/mvp_episodes_overnight/ \
    --progress
```

---

## Restore Data

### Restore from Backup

```bash
# Restore to local directory
rclone copy s3-salus-backup:salus-data-backup/ \
    /home/mpcr/Desktop/Salus\ Test/SalusTest/data-restored/ \
    --progress
```

### Verify Backup Integrity

```bash
# Check if local and remote match
rclone check /home/mpcr/Desktop/Salus\ Test/SalusTest/data \
    s3-salus-backup:salus-data-backup/ \
    --one-way
```

---

## Storage Cost Estimates

### For 500 Episodes (~50GB compressed):

| Service | Cost/Month (50GB) | Notes |
|---------|-------------------|-------|
| **AWS S3 Standard** | ~$1.15 | Fast, reliable |
| **AWS S3 Glacier** | ~$0.10 | Archive (slower retrieval) |
| **Google Cloud Storage** | ~$1.00 | Similar to S3 |
| **Backblaze B2** | ~$0.25 | Very cheap, fast |
| **Google Drive** | Free (15GB), $2/mo (100GB) | Good for small scale |

**Recommendation**: Backblaze B2 for cost, AWS S3 for reliability

---

## Best Practices

### 1. Backup Strategy

```
Daily:   Incremental backup (only new/modified files)
Weekly:  Full sync (verify integrity)
Monthly: Archive old data to Glacier/deep archive
```

### 2. Multiple Backups (3-2-1 Rule)

- **3 copies** of data (local + 2 remote)
- **2 different** storage types (S3 + B2)
- **1 offsite** (cloud is offsite by default)

### 3. Backup Before Major Operations

```bash
# Backup before training (protect collected data)
./scripts/backup_data.sh

# Then run training
python scripts/train_predictor_mvp.py
```

### 4. Exclude Unnecessary Files

```bash
# Only backup Zarr data, not logs/temp files
rclone sync "$LOCAL_DATA_DIR" "$REMOTE_NAME:$REMOTE_PATH/" \
    --include "**/data.zarr/**" \
    --exclude "*.log" \
    --exclude "*.tmp" \
    --exclude "__pycache__"
```

---

## Zarr-Specific Advantages

### Why Zarr + Rclone is Perfect:

1. **Chunked Structure**: Small files sync faster than one huge file
2. **Partial Recovery**: Can restore specific episodes without downloading all
3. **Parallel Uploads**: Multiple chunks upload simultaneously
4. **Resume Support**: Failed chunks retry individually
5. **Direct S3 Access**: Zarr can read directly from S3 (no download needed!)

### Future: Direct S3 Zarr Access

```python
# Zarr can read directly from S3 (future optimization)
import zarr
import s3fs

s3 = s3fs.S3FileSystem()
zarr_store = zarr.open(s3fs.S3Map('s3://salus-data-backup/data.zarr', s3=s3))

# Load data directly from S3 (no download!)
images = zarr_store['images'][:10]  # Load first 10 episodes
```

---

## Quick Start Guide

### 1. Install Rclone
```bash
curl https://rclone.org/install.sh | sudo bash
```

### 2. Configure Cloud Storage
```bash
rclone config
# Follow prompts to set up S3, GCS, or B2
```

### 3. Test Backup
```bash
# Dry run (see what would be backed up)
rclone sync data/ s3-salus-backup:salus-data-backup/ --dry-run

# Actual backup
rclone sync data/ s3-salus-backup:salus-data-backup/ --progress
```

### 4. Set Up Automated Backup
```bash
# Create backup script (see above)
# Add to crontab for daily backups
crontab -e
```

---

## Monitoring Backup Status

```bash
# Check backup size
rclone size s3-salus-backup:salus-data-backup/

# List backed up files
rclone ls s3-salus-backup:salus-data-backup/

# Show transfer statistics
rclone sync data/ s3-salus-backup:salus-data-backup/ --stats 10s
```

---

## Summary

✅ **YES - Use Rclone for SALUS data backup**

**Why**:
- Zarr format is cloud-native (many small files)
- Efficient incremental syncs
- Works with all major cloud providers
- Handles large datasets (50-100GB+)
- Can resume interrupted transfers

**Recommended Setup**:
- **Primary**: Backblaze B2 (cheap, $0.25/month for 50GB)
- **Secondary**: AWS S3 (reliable, $1.15/month for 50GB)
- **Automation**: Daily cron job for incremental backups

**Cost**: ~$1-2/month for 50GB backup (very affordable!)

---

**Next Steps**: Install Rclone, configure cloud storage, run first backup, set up automated daily backups.




