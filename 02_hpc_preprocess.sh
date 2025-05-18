#!/bin/bash
#SBATCH --job-name="preprocess"
#SBATCH --account=3189081
#SBATCH --partition=ai
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gpus=0
#SBATCH --output=output/%x_%j.out # %x gives job name and %j gives job id
#SBATCH --error=output/%x_%j.err

cd ~/PRJ/birdclef2025/

module load modules/miniconda3

eval "$(conda shell.bash hook)"

conda activate birdclef

python audio_to_mel.py