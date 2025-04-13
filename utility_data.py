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


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, datafolder:str="data", metadata_csv: str="train.csv", audio_dir: str="train_audio",
                 transform=None, metadata: bool=False, extract_features=False, audio_params: dict=None):
        """
        metadata_csv: path to train.csv
        audio_dir: path to train_audio/
        transform: transform for waveform
        extract_features: whether to exract rich feature set for CNN
        audio_params: parameters for audio features
        feature_size: target width for feature padding
        sample_rate: target sample rate for audio processing
        """
        datafolder = os.path.join(datafolder, "") 
        audio_dir = os.path.join(audio_dir, "")  
        self.audio_dir = os.path.join(datafolder, audio_dir) 
        self.transform = transform
        self.extract_features = extract_features
        params = [audio_params[k] for k in ["sample_rate", "n_fft", "hop_length", "n_mfcc", "n_mels", "feature_size"]]
        self.sample_rate, self.n_fft, self.hop_length, self.n_mfcc, self.n_mels, self.feature_size = params

        if metadata_csv == "":
            self.data = self._load_audio_data(self.audio_dir)
        else:
            self.data = pd.read_csv(os.path.join(datafolder, metadata_csv))
            
        if metadata:
            metadata_df = self.data["filename"].apply(
                lambda filename: self._get_audio_metadata(os.path.join(self.audio_dir, filename))
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

        label = self.class_to_idx[row["primary_label"]] if "primary_label" in row else -1

        if self.extract_features:
            try:
                # Extract rich feature set for CNN
                features = self.generate_features(audio_path)
                # Convert to torch tensor
                features_tensor = torch.tensor(features, dtype=torch.float32)
                features_tensor = features_tensor.permute(2, 0, 1)  # [channels, height, width]
                return features_tensor, label
            except Exception as e:
                print(f"Error extracting features from {audio_path}: {e}")
                # Return dummy data with appropriate shape
                return torch.zeros(4, 128, self.feature_size), label
        else:
            try:
                waveform, _ = torchaudio.load(audio_path)
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
        features = self._prepare_features(audio_path)
        label = self.class_to_idx[row["primary_label"]] if "primary_label" in row else -1
        return features, label

    def _get_audio_metadata(self, audio_path):
        """Extract metadata from audio file"""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            duration = waveform.shape[1] / sample_rate
            return {"duration": duration, "sample_rate": sample_rate, "channels": waveform.shape[0]}
        except Exception as e:
            print(f"Error loading metadata for {audio_path}: {e}")
            return {"duration": 0, "sample_rate": 0, "channels": 0}

    def _load_audio_data(self, audio_dir):
        """Create a DataFrame from audio files in a directory"""
        file_list = []
        for root, _, files in os.walk(audio_dir):
            for file in files:
                if file.endswith('.ogg') or file.endswith('.wav'):
                    rel_path = os.path.relpath(os.path.join(root, file), audio_dir)
                    file_list.append(rel_path)
        return pd.DataFrame({"filename": file_list})

    def _normalize(self, array):
        """Normalize array to [0, 1]"""
        min_val = np.min(array)
        max_val = np.max(array)
        if max_val > min_val:
            return (array - min_val) / (max_val - min_val)
        return np.zeros_like(array)

    def _padding(self, array, xx, yy):
        """
        Pad or trim array to specified dimensions
        :param array: numpy array
        :param xx: desired height
        :param yy: desired width
        :return: padded or trimmed array
        """
        h, w = array.shape[0], array.shape[1]
        
        # If array is already equal to or wider than target, just trim to target width
        if w >= yy:
            # Center the content when trimming
            start_idx = (w - yy) // 2
            return array[:, start_idx:start_idx+yy]
        
        # If array is narrower, pad to target width
        a = max((xx - h) // 2, 0)
        aa = max(0, xx - a - h)
        b = max(0, (yy - w) // 2)
        bb = max(yy - b - w, 0)
        
        return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')

    def generate_features(self, audio_path):
        try:
            # Load audio file
            y_cut, _ = librosa.load(audio_path, sr=self.sample_rate)
            
            # Check if audio loaded properly
            if np.all(y_cut == 0) or len(y_cut) == 0:
                print(f"Warning: Audio file {audio_path} loaded as silence or empty")
                return np.zeros((64, self.feature_size, 3))
                
            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y_cut, sr=self.sample_rate, n_fft=self.n_fft, 
                hop_length=self.hop_length, n_mels=self.n_mels,
                fmax=self.sample_rate/2 
            )
            mel_spec = self._normalize(np.abs(mel_spec))
            mel_spec = self._padding(mel_spec, self.n_mels, self.feature_size)
            
            # MFCCs
            MFCCs = librosa.feature.mfcc(
                y=y_cut, sr=self.sample_rate, n_fft=self.n_fft,
                hop_length=self.hop_length, n_mfcc=self.n_mfcc
            )
            MFCCs = self._normalize(MFCCs)
            MFCCs = self._padding(MFCCs, self.n_mfcc, self.feature_size)
            
            # Create third channel with combined spectral features
            spec_centroid = librosa.feature.spectral_centroid(
                y=y_cut, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
            )
            spec_centroid = self._padding(self._normalize(spec_centroid), 1, self.feature_size)
            
            # Chroma STFT (12 rows)
            chroma_stft = librosa.feature.chroma_stft(
                y=y_cut, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
            )
            chroma_stft = self._padding(self._normalize(chroma_stft), 12, self.feature_size)
            
            # Spectral bandwidth (1 row)
            spec_bw = librosa.feature.spectral_bandwidth(
                y=y_cut, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
            )
            spec_bw = self._padding(self._normalize(spec_bw), 1, self.feature_size)
            
            # Combine features for third channel
            channel3 = np.zeros((64, self.feature_size))
            
            # First add spectral bandwidth and centroid
            channel3[0:1] = spec_bw
            channel3[1:2] = spec_centroid
            
            # Add chroma (12 rows)
            channel3[2:14] = chroma_stft
            
            # Fill remaining rows by repeating spec_bw and spec_centroid
            current_row = 14
            while current_row < 64:
                if current_row % 2 == 0 and current_row + 1 <= 64:
                    channel3[current_row:current_row+1] = spec_bw
                elif current_row + 1 <= 64:
                    channel3[current_row:current_row+1] = spec_centroid
                current_row += 1
            
            # Make sure all features have the same shape (64, feature_size)
            assert channel3.shape == (64, self.feature_size)
            assert mel_spec.shape == (64, self.feature_size)
            assert MFCCs.shape == (64, self.feature_size)
            
            # Stack the features
            result = np.stack([channel3, mel_spec, MFCCs], axis=2)
            return result
            
        except Exception as e:
            print(f"Error in generating features for {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((64, self.feature_size, 3))