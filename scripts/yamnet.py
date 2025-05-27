import csv
import io
import os
import ast
import argparse

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import pandas as pd

import librosa

parser = argparse.ArgumentParser(description="YAMNet label extraction script")

parser.add_argument('--data', type=str, default='data/',
                    help='Root folder containing audio and metadata files (default: data/)')
parser.add_argument('--audios_folder', type=str, default='train_proc',
                    help='Subfolder within data/ containing the audio files (default: train_proc)')
parser.add_argument('--metadata_csv', type=str, default='train_proc.csv',
                    help='Metadata CSV file name in data/ (default: train_proc.csv)')
parser.add_argument('--output_file', type=str, default='train_proc_yamn.csv',
                    help='Output CSV filename to save labels (default: train_proc_yamn.csv)')

args = parser.parse_args()

# Construct full paths
data_folder = os.path.join(args.data, args.audios_folder)
data_file = os.path.join(args.data, args.metadata_csv)
output_file = os.path.join(args.data, args.output_file)

TARGET_SR = 16000  # Target sample rate


def load_resample(file_path):
    """Loads and resamples audio to TARGET_SR using librosa, returns TF-compatible array"""
    # Load audio with librosa (automatically converts to mono)
    audio, sr = librosa.load(file_path, sr=None, mono=True)    
    audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio.astype(np.float32)

# Offline Setup: mkdir data/yamnet and therein...
# curl -L -o model.tar.gz https://www.kaggle.com/api/v1/models/google/yamnet/tensorFlow2/yamnet/1/download
model = hub.load(os.path.join(args.data, 'yamnet'))

# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_map_csv = io.StringIO(class_map_csv_text)
  class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
  class_names = class_names[1:]  # Skip CSV header
  return class_names

def yamnet_classification(file_path):
    waveform = load_resample(file_path)
    scores, _, _ = model(waveform)

    class_map_path = model.class_map_path().numpy()
    class_names = class_names_from_csv(tf.io.read_file(class_map_path).numpy().decode('utf-8'))
    
    return class_names[scores.numpy().mean(axis=0).argmax()]


data = pd.read_csv(data_file)
data['filename'] = data['filename'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x)
data = data.explode('filename', ignore_index=True)

data['yamnet'] = data['filename'].apply(lambda path: yamnet_classification(os.path.join(os.getcwd(), data_folder, path)))

data.to_csv(output_file, index=False)

print('\n Yamnet Operation Concluded Successfully \n')