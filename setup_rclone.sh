#!/bin/bash
# Rclone Setup for Google Cloud Storage
# Run this after hpc_setup.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Rclone Setup for Google Cloud${NC}"
echo -e "${GREEN}========================================${NC}"

# Step 1: Install Rclone
echo -e "\n${YELLOW}Step 1: Installing Rclone...${NC}"

if command -v rclone &> /dev/null; then
    RCLONE_VERSION=$(rclone version | head -n 1)
    echo -e "  ${GREEN}✓ Rclone already installed: $RCLONE_VERSION${NC}"
else
    echo -e "  Installing Rclone..."
    curl https://rclone.org/install.sh | sudo bash
    echo -e "  ${GREEN}✓ Rclone installed${NC}"
fi

# Step 2: Configure Rclone
echo -e "\n${YELLOW}Step 2: Configuring Google Cloud Storage...${NC}"
echo -e ""
echo -e "${BLUE}Please follow these steps:${NC}"
echo -e "  1. Run: ${YELLOW}rclone config${NC}"
echo -e "  2. Choose: ${YELLOW}n) New remote${NC}"
echo -e "  3. Name: ${YELLOW}salus_gcloud${NC}"
echo -e "  4. Storage: ${YELLOW}Google Cloud Storage${NC} (find the number for 'gcs')"
echo -e "  5. For authentication, choose one of:"
echo -e "     ${YELLOW}Option A (Recommended):${NC} Service Account"
echo -e "       - Create service account in GCP console"
echo -e "       - Download JSON key"
echo -e "       - Provide path to JSON key file"
echo -e "     ${YELLOW}Option B:${NC} OAuth (opens browser)"
echo -e "       - Choose 'auto config'"
echo -e "       - Follow browser authentication"
echo -e "  6. Complete configuration"
echo -e ""
echo -e "${YELLOW}Press Enter when ready to run rclone config...${NC}"
read

rclone config

# Step 3: Create bucket
echo -e "\n${YELLOW}Step 3: Creating Google Cloud bucket...${NC}"
echo -e ""
read -p "Enter bucket name (e.g., salus-a100-data): " BUCKET_NAME

if rclone lsd salus_gcloud: 2>/dev/null | grep -q "$BUCKET_NAME"; then
    echo -e "  ${GREEN}✓ Bucket '$BUCKET_NAME' already exists${NC}"
else
    echo -e "  Creating bucket: $BUCKET_NAME"
    rclone mkdir salus_gcloud:$BUCKET_NAME
    echo -e "  ${GREEN}✓ Bucket created${NC}"
fi

# Step 4: Test connection
echo -e "\n${YELLOW}Step 4: Testing connection...${NC}"

echo "test file" > /tmp/rclone_test.txt
rclone copy /tmp/rclone_test.txt salus_gcloud:$BUCKET_NAME/test/
rm /tmp/rclone_test.txt

if rclone ls salus_gcloud:$BUCKET_NAME/test/ | grep -q "rclone_test.txt"; then
    echo -e "  ${GREEN}✓ Connection test successful!${NC}"
    rclone delete salus_gcloud:$BUCKET_NAME/test/rclone_test.txt
else
    echo -e "  ${RED}✗ Connection test failed${NC}"
    exit 1
fi

# Step 5: Create backup script
echo -e "\n${YELLOW}Step 5: Creating automated backup script...${NC}"

cat > ~/salus_a100/auto_backup.sh << 'BACKUP_SCRIPT'
#!/bin/bash
# Automated SALUS Data Backup to Google Cloud

set -e

# Configuration
LOCAL_BASE=~/salus_a100
REMOTE_BASE="salus_gcloud:BUCKET_NAME_PLACEHOLDER"
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
BACKUP_SCRIPT

# Replace bucket name placeholder
sed -i "s/BUCKET_NAME_PLACEHOLDER/$BUCKET_NAME/g" ~/salus_a100/auto_backup.sh
chmod +x ~/salus_a100/auto_backup.sh

echo -e "  ${GREEN}✓ Backup script created: ~/salus_a100/auto_backup.sh${NC}"

# Step 6: Setup cron job (optional)
echo -e "\n${YELLOW}Step 6: Setup automated backups (optional)${NC}"
read -p "Setup cron job to backup every 6 hours? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Add cron job
    (crontab -l 2>/dev/null; echo "0 */6 * * * $HOME/salus_a100/auto_backup.sh") | crontab -
    echo -e "  ${GREEN}✓ Cron job added (runs every 6 hours)${NC}"
else
    echo -e "  ${YELLOW}⚠ Skipped. Run manually: ~/salus_a100/auto_backup.sh${NC}"
fi

# Summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Rclone Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Configuration:${NC}"
echo -e "  Remote name: ${BLUE}salus_gcloud${NC}"
echo -e "  Bucket: ${BLUE}$BUCKET_NAME${NC}"
echo -e "  Backup script: ${BLUE}~/salus_a100/auto_backup.sh${NC}"

echo -e "\n${YELLOW}Usage:${NC}"
echo -e "  Test backup: ${BLUE}~/salus_a100/auto_backup.sh${NC}"
echo -e "  List bucket: ${BLUE}rclone ls salus_gcloud:$BUCKET_NAME${NC}"
echo -e "  Check size: ${BLUE}rclone size salus_gcloud:$BUCKET_NAME${NC}"

echo -e "\n${GREEN}Ready for data collection!${NC}"
