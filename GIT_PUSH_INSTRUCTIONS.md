# Git Push Instructions

## Current Status

Files are committed locally but need to be pushed to GitHub. The push failed due to authentication.

## Authentication Options

### Option 1: Personal Access Token (Recommended)

1. **Create a GitHub Personal Access Token**:
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Select scopes: `repo` (full control of private repositories)
   - Copy the token (you'll only see it once!)

2. **Push using token**:
   ```bash
   cd "/home/mpcr/Desktop/Salus Test/SalusTest"
   git push origin main
   # When prompted:
   # Username: AnikS22
   # Password: [paste your personal access token]
   ```

3. **Or configure credential helper** (to avoid entering token each time):
   ```bash
   git config --global credential.helper store
   git push origin main
   # Enter token once, it will be saved
   ```

### Option 2: Switch to SSH (More Secure)

1. **Check if you have SSH key**:
   ```bash
   ls -la ~/.ssh/id_rsa.pub
   ```

2. **If no SSH key, generate one**:
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   # Press Enter to accept defaults
   ```

3. **Add SSH key to GitHub**:
   ```bash
   cat ~/.ssh/id_rsa.pub
   # Copy the output
   # Go to: https://github.com/settings/keys
   # Click "New SSH key", paste the key
   ```

4. **Switch remote to SSH**:
   ```bash
   cd "/home/mpcr/Desktop/Salus Test/SalusTest"
   git remote set-url origin git@github.com:AnikS22/SalusTest.git
   git push origin main
   ```

### Option 3: GitHub CLI (gh)

```bash
# Install GitHub CLI
sudo apt install gh  # or: curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg

# Authenticate
gh auth login

# Push (will use gh authentication)
git push origin main
```

## Quick Commands

```bash
cd "/home/mpcr/Desktop/Salus Test/SalusTest"

# Check status
git status

# View commit
git log -1

# Push (after setting up authentication)
git push origin main
```

## What's Being Pushed

✅ **Included**:
- All Python code (salus/, scripts/)
- Documentation files (*.md)
- Configuration files (configs/)
- Setup scripts

❌ **Excluded** (via .gitignore):
- data/ (Zarr files, 20GB)
- models/ (trained models)
- checkpoints/ (model checkpoints)
- logs/ (log files)
- venv_salus/ (virtual environment)

## Current Commit

The commit includes:
- 27 Python files (implementation code)
- 23 documentation files
- Configuration files
- Updated setup script

Total: ~4,175 lines of code + comprehensive documentation




