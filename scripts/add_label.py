from efficient_net import EfficientNetAudio
from utility_data import AudioDataset

import torch
import torchaudio

import numpy as np
import pandas as pd

import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Efficient label extraction script")
parser.add_argument('--data', type=str, default='data/',
                    help='Root folder containing audio and metadata files (default: data/)')

parser.add_argument('--audios_folder', type=str, default='train_proc',
                    help='Subfolder within data/ containing the audio files (default: train_proc)')
parser.add_argument('--metadata_csv', type=str, default='train_proc.csv',
                    help='Metadata CSV file name in data/ (default: train_proc.csv)')

parser.add_argument('--model_folder', type=str, default='model/',
                    help='Root folder containing all models (default: model/)')

parser.add_argument('--model_name', type=str,
                    help='Root folder containing audio and metadata files')

parser.add_argument('--checkpoint', type=str,
                    help='Name of the checkpoint .ckpt file in the "checkpoints/" subfolder within model_folder')

parser.add_argument('--output_file', type=str, default='train_proc_efficient.csv',
                    help='Output CSV filename to save labels (default: train_proc_yamn.csv)')

args = parser.parse_args()

print(args)

output_file = os.path.join(args.data, args.output_file)
model_path = os.path.join(args.model_folder, args.model_name, 'checkpoints', args.checkpoint)

assert Path(model_path).exists()

# load the data into memory
dataset = AudioDataset(
    datafolder=args.data,
    metadata_csv=args.metadata_csv,
    audio_dir=args.audios_folder,
    feature_mode='mel'
)

model = EfficientNetAudio.load_from_checkpoint(model_path)
model.eval()

def classification(waveform):
    x = waveform

    with torch.no_grad():
        y = model(x)
        return torch.argmax(y, dim=1)


def sample(idx: int):
    waveform = dataset.get_feature(idx)
    waveform = waveform.clone().detach().unsqueeze(0)
    return int(classification(waveform))
    
dataset.data[model_path] = dataset.data.index.map(sample)

dataset.data.to_csv(output_file)

print('\n Operation Concluded Successfully \n')