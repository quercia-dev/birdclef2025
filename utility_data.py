import os
import pandas as pd
import torch
import sys
import subprocess
import torch
import torchaudio
import librosa
import librosa.display
import numpy as np

from mutagen.oggvorbis import OggVorbis
from tinytag import TinyTag
from torchaudio.transforms import MelSpectrogram, MFCC
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, datafolder:str="data", metadata_csv: str="train.csv", audio_dir: str="train_audio",
                 transform: MelSpectrogram=None, metadata: bool=False, extract_features=False):
        """
        metadata_csv: path to train.csv
        audio_dir: path to train_audio/
        transform: transform for waveform
        extract_features: whether to exract rich feature set for CNN
        """
        datafolder = os.path.join(datafolder, "") 
        audio_dir = os.path.join(audio_dir, "")  
        self.audio_dir = os.path.join(datafolder, audio_dir) 
        self.transform = transform
        self.extract_features = extract_features

        if metadata_csv == "":
            self.data = load_audio_data(self.audio_dir)
        else:
            self.data = pd.read_csv(os.path.join(datafolder, metadata_csv))
            
        if metadata:
            metadata_df = self.data["filename"].apply(
                lambda filename: get_audio_metadata(os.path.join(self.audio_dir, filename))
            )
            # Unpack the dictionary and assign to new columns
            metadata_df = pd.DataFrame(metadata_df.tolist())
            self.data = pd.concat([self.data, metadata_df], axis=1)

        # sort by alphabetical order, then map species name to label index
        self.classes = sorted(self.data["primary_label"].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get metadata row of specified index
        row = self.data.iloc[idx]

        # construct path to audio file
        audio_path = os.path.join(self.audio_dir, row["filename"])

        label = self.class_to_idx[row["primary_label"]]

        if self.extract_features:
            try:
                # Extract rich feature set for CNN
                features = prepare_features(audio_path)
                # Convert to torch tensor
                features_tensor = torch.tensor(features, dtype=torch.float32)
                return features_tensor, label
            except Exception as e:
                print(f"Error extracting features from {audio_path}: {e}")
                # Return dummy data with appropriate shape
                return torch.zeros(128, 801, 4), label
        else:
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                # apply any transformation if specified
                if self.transform:
                    waveform = self.transform(waveform)
                return waveform, label
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                return torch.zeros(1, 16000), label  # dummy data if missing file

    def open(self, idx):
        row = self.data.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row["filename"])

        if sys.platform == 'win32':  # For Windows
            os.startfile(audio_path)
        elif sys.platform == 'darwin':  # For macOS
            subprocess.run(['open', audio_path])
        elif sys.platform == 'linux':  # For Linux
            subprocess.run(['xdg-open', audio_path])
        else:
            print(f"Unsupported OS: {sys.platform}")

    def locate(self, idx):
        """Open the folder and select the file."""
        row = self.data.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row["filename"])  

        if sys.platform == 'win32':  # For Windows
            subprocess.run(['explorer', '/select,', audio_path])
        elif sys.platform == 'darwin':  # For macOS
            subprocess.run(['open', '-R', audio_path])
        elif sys.platform == 'linux':  # For Linux
            subprocess.run(['nautilus', '--select', audio_path])
        else:
            print(f"Unsupported OS: {sys.platform}")

    def get_features(self, idx):
        """Extract and return features for a specific sample"""
        row = self.data.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row["filename"])
        return prepare_features(audio_path), self.class_to_idx[row["primary_label"]]


def get_audio_metadata(audio_path):
    """Extract metadata from audio file"""
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        duration = waveform.shape[1] / sample_rate
        return {"duration": duration, "sample_rate": sample_rate, "channels": waveform.shape[0]}
    except Exception as e:
        print(f"Error loading metadata for {audio_path}: {e}")
        return {"duration": 0, "sample_rate": 0, "channels": 0}

def load_audio_data(audio_dir):
    """Create a DataFrame from audio files in a directory"""
    file_list = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.ogg') or file.endswith('.wav'):
                rel_path = os.path.relpath(os.path.join(root, file), audio_dir)
                file_list.append(rel_path)
    return pd.DataFrame({"filename": file_list})


def normalize(array):
    """Normalize array to [0, 1]"""
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val > min_val:
        return (array - min_val) / (max_val - min_val)
    return np.zeros_like(array)


def padding(array, xx, yy):
    """
    Pad array to specified dimensions
    :param array: numpy array
    :param xx: desired height
    :param yy: desired width
    :return: padded array
    """
    
    h = array.shape[0]
    w = array.shape[1]
    
    a = max((xx - h) // 2, 0)
    aa = max(0, xx - a - h)
    b = max(0, (yy - w) // 2)
    bb = max(yy - b - w, 0)
    
    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


def prepare_features(audio_path: str, sample_rate: int=32000, max_size: int=801):
    """
    Load audio file and extract features
    :param max_size: target width for feature padding
    :return: processed feature image
    """
    y, sr = librosa.load(audio_path, sr=sample_rate)

    # extract features

    # Short-time fourier transform
    stft = np.abs(librosa.stft(y, n_fft=255, hop_length=512))
    stft_padded = padding(stft, 128, max_size)

    #mel spectrogram
    mel_spec = librosa.feature.mfcc(snippet_np, n_fft=255, hop_length=512, n_mfcc=128)
    mel_padded = padding(mel_spec, 128, max_size)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_fft=255, hop_length=512, n_mfcc=128)
    mfccs_padded = padding(mfccs, 128, max_size)
    
    # spectral centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spec_centroid_norm = normalize(spec_centroid)
    
    # chroma features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_norm = normalize(chroma_stft)
    
    # spectral bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spec_bw_norm = normalize(spec_bw)

    # build image layers
    image = np.array([padding(spec_bw_norm.reshape(1, -1), 1, max_size)]).reshape(1, max_size)
    image = np.append(image, padding(spec_centroid_norm.reshape(1, -1), 1, max_size), axis=0)

    # repeat padded features
    for i in range(0, 9):
        image = np.append(image, padding(spec_bw_norm.reshape(1, -1), 1, max_size), axis=0)
        image = np.append(image, padding(spec_centroid_norm.reshape(1, -1), 1, max_size), axis=0)
        image = np.append(image, padding(chroma_stft_norm, 12, max_size), axis=0)
    
    image = np.dstack((image, stft_padded))
    image = np.dstack((image, mfccs_padded))
    image = np.dstack((image, mel_padded))

    return image