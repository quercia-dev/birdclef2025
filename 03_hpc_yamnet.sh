#!/bin/bash
#SBATCH --job-name="train"
#SBATCH --account=3189081
#SBATCH --partition=ai
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --gpus=0
#SBATCH --output=output/%x_%j.out # %x gives job name and %j gives job id
#SBATCH --error=output/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=3189081@studbocconi.it

cd ~/PRJ/birdclef2025/

module load modules/miniconda3

eval "$(conda shell.bash hook)"

conda activate yamnet

# label the training data
python yamnet.py --audios_folder train_proc --metadata_csv train_proc.csv --output_file train_proc_yamn.csv

# label the soundscapes data
python yamnet.py --audios_folder train_soundscapes_proc --metadata_csv train_soundscapes_proc.csv --output_file train_soundscapes_proc_yamn.csv