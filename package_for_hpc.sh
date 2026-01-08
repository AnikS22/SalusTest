#!/bin/bash
# Package SALUS for HPC Deployment
# Run this on your local machine before transferring to HPC

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Packaging SALUS for HPC Deployment${NC}"
echo -e "${GREEN}========================================${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create package directory
PACKAGE_NAME="salus_a100_deployment_$(date +%Y%m%d_%H%M%S)"
echo -e "\n${YELLOW}Creating package: $PACKAGE_NAME${NC}"

mkdir -p "$PACKAGE_NAME"

# Copy necessary files
echo -e "${YELLOW}Copying files...${NC}"

# Core SALUS code
echo "  - salus/ (core modules)"
cp -r salus "$PACKAGE_NAME/"

# Scripts
echo "  - scripts/ (data collection, training)"
cp -r scripts "$PACKAGE_NAME/"

# Configs
echo "  - configs/ (A100 configuration)"
cp -r configs "$PACKAGE_NAME/"

# Deployment scripts
echo "  - Deployment scripts"
cp hpc_setup.sh "$PACKAGE_NAME/"
cp setup_rclone.sh "$PACKAGE_NAME/"
cp test_salus.sh "$PACKAGE_NAME/"
cp deploy_a100.sh "$PACKAGE_NAME/"

# Documentation
echo "  - Documentation"
cp HPC_DEPLOYMENT.md "$PACKAGE_NAME/"
cp A100_SCALING_GUIDE.md "$PACKAGE_NAME/"
cp DATA_MANAGEMENT_GUIDE.md "$PACKAGE_NAME/"

# Make scripts executable
chmod +x "$PACKAGE_NAME"/*.sh

# Create README
cat > "$PACKAGE_NAME/README.md" << 'EOF'
# SALUS A100 Deployment Package

This package contains everything needed to deploy SALUS on an A100 HPC.

## Quick Start

1. **Transfer to HPC:**
   ```bash
   scp -r salus_a100_deployment_* username@hpc_address:~/
   ssh username@hpc_address
   cd salus_a100_deployment_*
   ```

2. **Run setup:**
   ```bash
   bash hpc_setup.sh
   bash setup_rclone.sh
   bash test_salus.sh
   ```

3. **Deploy:**
   ```bash
   bash deploy_a100.sh collect 500 8
   ```

## Documentation

- **HPC_DEPLOYMENT.md** - Complete deployment guide
- **A100_SCALING_GUIDE.md** - Technical details and optimization
- **DATA_MANAGEMENT_GUIDE.md** - Rclone backup strategies

## Support

If you encounter issues, check:
1. HPC_DEPLOYMENT.md - Troubleshooting section
2. Log files in a100_logs/
3. Test with 1 episode first: `bash test_salus.sh`

## Timeline

- Setup: 15 minutes
- Test: 13 minutes
- Data collection: ~50 hours (500 episodes)
- Training: 15 minutes
- **Total: ~51 hours**

Expected results: F1 > 0.60 (2× better than PoC)
EOF

# Create archive
echo -e "\n${YELLOW}Creating tar.gz archive...${NC}"
tar -czf "${PACKAGE_NAME}.tar.gz" "$PACKAGE_NAME/"

# Calculate size
PACKAGE_SIZE=$(du -sh "${PACKAGE_NAME}.tar.gz" | cut -f1)

# Cleanup temp directory
rm -rf "$PACKAGE_NAME"

# Summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Package Created Successfully!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Package Details:${NC}"
echo -e "  File: ${BLUE}${PACKAGE_NAME}.tar.gz${NC}"
echo -e "  Size: ${BLUE}$PACKAGE_SIZE${NC}"
echo -e "  Location: ${BLUE}$(pwd)/${PACKAGE_NAME}.tar.gz${NC}"

echo -e "\n${YELLOW}Contents:${NC}"
echo -e "  • salus/ - Core SALUS modules"
echo -e "  • scripts/ - Data collection and training scripts"
echo -e "  • configs/ - A100 configuration"
echo -e "  • Deployment scripts (hpc_setup.sh, deploy_a100.sh, etc.)"
echo -e "  • Documentation (guides and READMEs)"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "  1. Transfer to HPC:"
echo -e "     ${BLUE}scp ${PACKAGE_NAME}.tar.gz username@hpc_address:~/${NC}"
echo -e ""
echo -e "  2. SSH into HPC:"
echo -e "     ${BLUE}ssh username@hpc_address${NC}"
echo -e ""
echo -e "  3. Extract and setup:"
echo -e "     ${BLUE}tar -xzf ${PACKAGE_NAME}.tar.gz${NC}"
echo -e "     ${BLUE}cd ${PACKAGE_NAME}${NC}"
echo -e "     ${BLUE}bash hpc_setup.sh${NC}"
echo -e ""
echo -e "  4. Follow instructions in HPC_DEPLOYMENT.md"

echo -e "\n${GREEN}Ready for HPC deployment!${NC}"
