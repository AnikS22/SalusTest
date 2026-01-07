#!/bin/bash
# Setup rclone on HPC for data backup
# Run this ON THE HPC CLUSTER

set -e

echo "============================================================"
echo "rclone Setup for HPC Data Backup"
echo "============================================================"
echo ""

# Install rclone to ~/bin if not already installed
if [ -f "$HOME/bin/rclone" ]; then
    echo "✓ rclone already installed at ~/bin/rclone"
else
    echo "Installing rclone to ~/bin/..."

    cd /tmp
    curl -O https://downloads.rclone.org/rclone-current-linux-amd64.zip
    unzip -q rclone-current-linux-amd64.zip
    cd rclone-*-linux-amd64

    mkdir -p ~/bin
    cp rclone ~/bin/
    chmod +x ~/bin/rclone

    # Add to PATH if not already there
    if ! grep -q "export PATH=\$HOME/bin:\$PATH" ~/.bashrc; then
        echo 'export PATH=$HOME/bin:$PATH' >> ~/.bashrc
        echo "✓ Added ~/bin to PATH in ~/.bashrc"
    fi

    cd ~
    rm -rf /tmp/rclone-*

    echo "✓ rclone installed successfully"
fi

echo ""
echo "rclone version:"
~/bin/rclone version | head -3
echo ""

echo "============================================================"
echo "Configure rclone Remote"
echo "============================================================"
echo ""
echo "You need to configure a remote storage location."
echo "Options:"
echo "  1. Google Drive (recommended - free 15GB)"
echo "  2. Dropbox"
echo "  3. OneDrive"
echo "  4. Amazon S3"
echo "  5. Your local machine (via SFTP)"
echo ""
echo "Run: ~/bin/rclone config"
echo ""
echo "Follow prompts to add a remote named 'backup'"
echo ""
echo "Example for Google Drive:"
echo "  n) New remote"
echo "  name> backup"
echo "  Storage> drive"
echo "  client_id> (leave empty)"
echo "  client_secret> (leave empty)"
echo "  scope> 1"
echo "  root_folder_id> (leave empty)"
echo "  service_account_file> (leave empty)"
echo "  Edit advanced config? n"
echo "  Use auto config? n"
echo "  [Follow URL to authorize]"
echo "  Configure this as a team drive? n"
echo ""
echo "After configuration, test with:"
echo "  ~/bin/rclone lsd backup:"
echo ""
