#!/bin/bash

cd ~/projects/birdclef2025/

# Load conda manually if not loaded already
source ~/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate birdclef

# Run your script
python utility_data.py
