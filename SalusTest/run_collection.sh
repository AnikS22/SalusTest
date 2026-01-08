#!/bin/bash
# Wrapper script to run data collection with Isaac Lab environment

# Activate isaaclab conda environment
source ~/miniconda/etc/profile.d/conda.sh
conda activate isaaclab

# Navigate to SalusTest directory
cd "/home/mpcr/Desktop/Salus Test/SalusTest"

# Run collection
python collect_local_data.py
