import csv
import io
import os
import ast

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import pandas as pd

import soundfile
import librosa

data_folder = "data/train_proc"
data_file = "data/train_proc.csv"
output_file = "data/train_proc_yamn.csv"
TARGET_SR = 16000  # Target sample rate

def load_resample(file_path):
    """Loads and resamples audio to TARGET_SR using librosa, returns TF-compatible array"""
    # Load audio with librosa (automatically converts to mono)
    audio, sr = librosa.load(file_path, sr=None, mono=True)    
    audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio.astype(np.float32)


model = hub.load('https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1')

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