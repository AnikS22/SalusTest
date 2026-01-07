# SALUS HPC Data Backup Strategy

**CRITICAL**: HPC storage is temporary! You MUST backup your data.

## The Problem

After 40-50 hours of data collection, you'll have:
- **~10-15GB** of training data (500 episodes)
- **~200KB-1MB** of model checkpoints
- **~10-50MB** of training logs

**HPC storage risks**:
- ⚠️ **Temporary storage** - Gets wiped after job completion
- ⚠️ **Storage quotas** - May hit limits
- ⚠️ **No backups** - Cluster crash = data loss
- ⚠️ **Auto-cleanup** - Old files deleted after 30-90 days

**Solution**: Backup data DURING and AFTER collection!

---

## Backup Strategy: Three Options

### Option 1: Cloud Backup (Recommended) ⭐

**Best for**: Long-term storage, multiple backups, accessibility

**Setup** (on HPC):
```bash
# 1. Install rclone
bash setup_rclone_hpc.sh

# 2. Configure cloud remote (Google Drive, Dropbox, etc.)
~/bin/rclone config
# Follow prompts to add remote named 'backup'

# 3. Test connection
~/bin/rclone lsd backup:
```

**Backup data**:
```bash
# On HPC, backup data to cloud
bash backup_from_hpc.sh
```

**Pros**:
- ✅ Automatic backups
- ✅ Multiple providers (Google Drive, Dropbox, S3)
- ✅ Accessible from anywhere
- ✅ Can resume interrupted transfers

**Cons**:
- ❌ Requires cloud account (but many have free tiers)
- ❌ Initial setup needed

---

### Option 2: Sync to Local Machine

**Best for**: If you have local storage and stable connection

**Setup** (on local machine):
```bash
export HPC_HOST=athene.informatik.tu-darmstadt.de
export HPC_USER=your_username
```

**Sync data back**:
```bash
# On local machine
./sync_from_hpc.sh
```

**Pros**:
- ✅ No cloud account needed
- ✅ Direct control of data
- ✅ Fast access locally

**Cons**:
- ❌ Requires stable internet (10-15GB transfer)
- ❌ Takes 30-60 min depending on connection
- ❌ Must manually run sync

---

### Option 3: Hybrid (Best Practice) 🌟

**Use BOTH cloud and local backup!**

1. **During training**: Backup to cloud periodically
2. **After training**: Sync to local for fast access
3. **Keep both**: Cloud as disaster recovery

---

## When to Backup

### Critical Backup Points

1. **After data collection completes** (MOST IMPORTANT!)
   ```bash
   # On HPC, immediately after 500 episodes collected
   bash backup_from_hpc.sh
   ```

2. **After training completes**
   ```bash
   # Backup checkpoints
   bash backup_from_hpc.sh
   ```

3. **Periodically during long runs** (every 12-24 hours)
   ```bash
   # Add to cron or run manually
   bash backup_from_hpc.sh
   ```

### Automated Backups (Optional)

Add to crontab on HPC:
```bash
# Backup every 6 hours
crontab -e

# Add line:
0 */6 * * * cd ~/SalusTest && bash backup_from_hpc.sh >> backup.log 2>&1
```

---

## Step-by-Step Guide

### Setup Phase (Do Once)

#### Cloud Backup Setup (Recommended)

**On HPC**:
```bash
# 1. Transfer backup scripts to HPC
# (done automatically by sync_to_hpc.sh)

# 2. SSH to HPC
ssh your_username@athene.informatik.tu-darmstadt.de
cd ~/SalusTest

# 3. Install rclone
bash setup_rclone_hpc.sh

# 4. Configure rclone for Google Drive
~/bin/rclone config

# Follow these prompts:
# n) New remote
# name> backup
# Storage> drive  (or your preferred service)
# [Follow authorization steps]

# 5. Test connection
~/bin/rclone lsd backup:
# Should show: OK if connected
```

**Google Drive Authorization**:
Since HPC has no browser, use this workflow:
1. rclone will give you a URL
2. Open URL on your local machine
3. Authorize and copy the code
4. Paste code back in HPC terminal

---

### Backup Workflows

#### After Data Collection (CRITICAL!)

```bash
# On HPC, after collection completes
cd ~/SalusTest

# Check data size
du -sh ~/salus_data_temporal/
# Should see: ~10-15GB

# Backup to cloud
bash backup_from_hpc.sh

# Monitor backup (in another terminal)
watch -n 10 '~/bin/rclone size backup:SALUS_HPC_Backup'
```

**Expected**:
- Transfer time: 30-60 min (depends on HPC network)
- Progress updates every 10 seconds
- Final confirmation when complete

#### After Training

```bash
# On HPC, after training completes
cd ~/SalusTest

# Check checkpoint size
du -sh checkpoints/temporal_baseline/
# Should see: ~1-5MB

# Backup (quick!)
bash backup_from_hpc.sh
```

#### Sync to Local (Optional but Recommended)

```bash
# On LOCAL MACHINE
cd /home/mpcr/Desktop/Salus\ Test/SalusTest

export HPC_HOST=athene.informatik.tu-darmstadt.de
export HPC_USER=your_username

# Sync data back
./sync_from_hpc.sh
```

**This will**:
- Show remote sizes
- Ask for confirmation
- Download to: `~/salus_data_hpc/`
- Download checkpoints to: `~/SalusTest/checkpoints_hpc/`

---

## Verification

### Check Cloud Backup

```bash
# On HPC
~/bin/rclone ls backup:SALUS_HPC_Backup/data/
~/bin/rclone ls backup:SALUS_HPC_Backup/checkpoints/

# Check total size
~/bin/rclone size backup:SALUS_HPC_Backup/
```

### Check Local Backup

```bash
# On local machine
ls -lh ~/salus_data_hpc/
ls -lh ~/SalusTest/checkpoints_hpc/

du -sh ~/salus_data_hpc/
```

---

## Recovery Procedures

### Restore from Cloud

```bash
# On HPC (if data lost)
~/bin/rclone sync backup:SALUS_HPC_Backup/data ~/salus_data_temporal
~/bin/rclone sync backup:SALUS_HPC_Backup/checkpoints ~/SalusTest/checkpoints
```

### Restore from Local

```bash
# From local to HPC
rsync -avz ~/salus_data_hpc/ your_username@athene.informatik.tu-darmstadt.de:~/salus_data_temporal/
```

---

## Storage Requirements

### Cloud Storage

**Minimum**: 15GB (covers training data + checkpoints)

**Free tier options**:
- Google Drive: 15GB free
- Dropbox: 2GB free (not enough, need paid)
- OneDrive: 5GB free (not enough)
- Mega.nz: 20GB free ⭐

**Recommended**: Google Drive (15GB free is perfect!)

### Local Storage

**Minimum**: 20GB (includes data + checkpoints + logs)

**Recommended**: 50GB (room for multiple runs)

---

## Backup Checklist

### Before Starting Data Collection

- [ ] rclone installed on HPC
- [ ] Cloud remote configured and tested
- [ ] Backup scripts transferred to HPC
- [ ] Local storage has 20GB+ free space

### During Data Collection

- [ ] Backup data every 12-24 hours (optional but safe)
- [ ] Monitor HPC storage quota: `quota -s`

### After Data Collection (CRITICAL!)

- [ ] Backup data to cloud immediately
- [ ] Verify backup: `~/bin/rclone ls backup:SALUS_HPC_Backup/data/`
- [ ] Sync to local machine (optional)
- [ ] Confirm data integrity (check file sizes match)

### After Training

- [ ] Backup checkpoints to cloud
- [ ] Backup training logs
- [ ] Sync to local machine
- [ ] Keep cloud backup for disaster recovery

---

## Quick Commands Reference

### On HPC

```bash
# Setup rclone (once)
bash setup_rclone_hpc.sh
~/bin/rclone config

# Backup everything
bash backup_from_hpc.sh

# Check backup
~/bin/rclone size backup:SALUS_HPC_Backup/

# Manual backup (if script fails)
~/bin/rclone sync ~/salus_data_temporal backup:SALUS_HPC_Backup/data/
```

### On Local Machine

```bash
# Sync from HPC to local
export HPC_HOST=athene.informatik.tu-darmstadt.de
export HPC_USER=your_username
./sync_from_hpc.sh

# Check local backup
du -sh ~/salus_data_hpc/
ls -lh ~/SalusTest/checkpoints_hpc/
```

---

## Troubleshooting

### "rclone not found"

```bash
# On HPC
bash setup_rclone_hpc.sh
source ~/.bashrc
~/bin/rclone version
```

### "Cannot connect to remote"

```bash
# Test configuration
~/bin/rclone config show backup

# Reconfigure
~/bin/rclone config

# Test connection
~/bin/rclone lsd backup:
```

### "Transfer too slow"

```bash
# Use more parallel transfers
~/bin/rclone sync ~/salus_data_temporal backup:SALUS_HPC_Backup/data/ \
    --transfers 8 \
    --checkers 16
```

### "Quota exceeded"

**On HPC**:
```bash
# Check quota
quota -s

# Clean up old data
rm -rf ~/old_project_data/
```

**On Cloud**:
- Upgrade to paid plan (Google Drive: 100GB = $2/month)
- Use multiple accounts
- Compress data before upload

---

## Best Practices

1. **Backup IMMEDIATELY after data collection** - Don't wait!
2. **Verify backups before deleting HPC data** - Check sizes match
3. **Use cloud + local** - Double protection
4. **Test recovery procedures** - Make sure you can restore
5. **Monitor storage quotas** - On both HPC and cloud
6. **Document backup locations** - Know where everything is
7. **Keep training logs** - Useful for debugging

---

## Cost Estimates

### Cloud Storage (for 15GB)

| Provider | Free Tier | Paid (100GB) |
|----------|-----------|--------------|
| Google Drive | 15GB (FREE ✅) | $2/month |
| Dropbox | 2GB | $10/month |
| OneDrive | 5GB | $2/month |
| Mega.nz | 20GB (FREE ✅) | $5/month |
| Amazon S3 | 5GB (1yr) | ~$2/month |

**Recommendation**: Google Drive or Mega.nz (both free!)

### Transfer Time

| Connection | 15GB Upload | 15GB Download |
|------------|-------------|---------------|
| 10 Mbps | ~3-4 hours | ~3-4 hours |
| 100 Mbps | ~20-30 min | ~20-30 min |
| 1 Gbps | ~2-3 min | ~2-3 min |

HPC typically has fast connections (100 Mbps - 1 Gbps).

---

## Emergency: Data Loss Prevention

**If HPC crashes or data deleted**:

1. **DON'T PANIC** - If you backed up to cloud, you're safe
2. **Check cloud backup**: `~/bin/rclone ls backup:SALUS_HPC_Backup/`
3. **Restore**: `~/bin/rclone sync backup:SALUS_HPC_Backup/data ~/salus_data_temporal`
4. **Resume training** - You didn't lose anything!

**If no backup exists**: You'll need to recollect data (40-50 hours) 😢

**Prevention**: ALWAYS backup after data collection completes!

---

## Summary

| Task | Command | When | Where |
|------|---------|------|-------|
| Setup rclone | `bash setup_rclone_hpc.sh` | Once | HPC |
| Backup data | `bash backup_from_hpc.sh` | After collection | HPC |
| Sync to local | `./sync_from_hpc.sh` | Optional | Local |
| Check backup | `~/bin/rclone size backup:SALUS_HPC_Backup/` | Verify | HPC |

**Golden Rule**: Backup BEFORE deleting HPC data!

---

**Last Updated**: January 6, 2026
**Status**: ✅ Scripts ready, instructions complete
**Next**: Setup rclone on HPC and test backup
