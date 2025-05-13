#!/bin/bash
#SBATCH --job-name="train"
#SBATCH --account=3189081
#SBATCH --partition=ai
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=output/%x_%j.out # %x gives job name and %j gives job id
#SBATCH --error=output/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=3189081@studbocconi.it

cd ~/PRJ/birdclef2025/

module load modules/miniconda3

eval "$(conda shell.bash hook)"

conda activate birdclef

python model.py --model efficient --epochs 10 --dataset yamnet_light --mass 1.0
python model.py --model efficient --epochs 10 --dataset yamnet_light --mass 0.8