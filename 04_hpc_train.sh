#!/bin/bash
#SBATCH --job-name="train"
#SBATCH --account=3189081
#SBATCH --partition=ai
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --output=output/%x_%j.out # %x gives job name and %j gives job id
#SBATCH --error=output/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=3189081@studbocconi.it

cd ~/PRJ/birdclef2025/

module load modules/miniconda3

eval "$(conda shell.bash hook)"

conda activate birdclef

python efficient_b_train.py