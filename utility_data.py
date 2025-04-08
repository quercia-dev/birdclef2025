import os
import pandas as pd
import torch
import sys

from mutagen.oggvorbis import OggVorbis
from tinytag import TinyTag
from torchaudio.transforms import MelSpectrogram, MFCC
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, datafolder:str="data", metadata_csv: str="train.csv", audio_dir: str="train_audio", transform: MelSpectrogram=None, metadata: bool=False):
        """
        metadata_csv: path to train.csv
        audio_dir: path to train_audio/
        transform: transform for waveform
        """
        datafolder = os.path.join(datafolder, "")  # Use os.path.join for cross-platform path handling
        audio_dir = os.path.join(audio_dir, "")  # Ensure the trailing slash is properly handled
        
        self.audio_dir = os.path.join(datafolder, audio_dir)  # Correct path joining
        self.transform = transform
        
        if metadata_csv == "":
            self.data = load_audio_data(self.audio_dir)
        else:
            self.data = pd.read_csv(os.path.join(datafolder, metadata_csv))  # Use os.path.join for metadata CSV path
            
        if metadata:
            metadata_df = self.data["filename"].apply(lambda filename: get_audio_metadata(os.path.join(self.audio_dir, filename)))  # Correct path joining
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
        audio_path = os.path.join(self.audio_dir, row["filename"])  # Correct path joining
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except:
            print(f"Error loading {audio_path}")
            return torch.zeros(1, 16000), -1 # dummy data if missing file

        # apply any transformation if specified
        if self.transform:
            waveform = self.transform(waveform)

        # get label
        label = self.class_to_idx[row["primary_label"]]

        return waveform, label
    
    def open(self, idx):
        row = self.data.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row["filename"])  # Correct path joining
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
        audio_path = os.path.join(self.audio_dir, row["filename"])  # Correct path joining

        if sys.platform == 'win32':  # For Windows
            subprocess.run(['explorer', '/select,', audio_path])
        elif sys.platform == 'darwin':  # For macOS
            subprocess.run(['open', '-R', audio_path])
        elif sys.platform == 'linux':  # For Linux
            subprocess.run(['nautilus', '--select', audio_path])
        else:
            print(f"Unsupported OS: {sys.platform}")


def get_audio_metadata(file_path: str) -> dict:
    """Extracts all audio metadata at once (file size, duration, bitrate, etc.)."""
    tag = TinyTag.get(file_path)
    
    file_size = os.path.getsize(file_path)
    duration = tag.duration  # Duration in seconds
    audio = OggVorbis(file_path)
    bitrate = audio.info.bitrate // 1000  # Convert from bps to kbps
    sample_rate = audio.info.sample_rate
    channels = audio.info.channels
    codec = audio.mime[0] if audio.mime else "Unknown"

    return {
        "file_size": file_size,
        "audio_duration": duration,
        "audio_bitrate": bitrate,
        "audio_sample_rate": sample_rate,
        "audio_channels": channels,
        "audio_codec": codec
    }

def load_audio_data(audio_dir: str) -> pd.DataFrame:
    """Constructs a DataFrame from the files 
    in the audio_dir path"""
    
    data_rows = []  # temp list
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(".ogg"):
                primary_label = os.path.splitext(file)[0]
                data_rows.append({
                                    "primary_label": primary_label, 
                                    "filename": file
                                    })
    
    return pd.DataFrame(data_rows)
