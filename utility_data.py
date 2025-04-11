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
                 transform=None, metadata: bool=False, extract_features=False,
                 feature_size: int=316, sample_rate: int=32000):
        """
        metadata_csv: path to train.csv
        audio_dir: path to train_audio/
        transform: transform for waveform
        extract_features: whether to exract rich feature set for CNN
        feature_size: target width for feature padding
        sample_rate: target sample rate for audio processing
        """
        datafolder = os.path.join(datafolder, "") 
        audio_dir = os.path.join(audio_dir, "")  
        self.audio_dir = os.path.join(datafolder, audio_dir) 
        self.transform = transform
        self.extract_features = extract_features
        self.feature_size = feature_size
        self.sample_rate = sample_rate

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

        label = self.class_to_idx[row["primary_label"]] if "primary_label" in row else -1

        if self.extract_features:
            try:
                # Extract rich feature set for CNN
                features = self._prepare_features(audio_path)
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
        features = self._prepare_features(audio_path)
        label = self.class_to_idx[row["primary_label"]] if "primary_label" in row else -1
        return features, label

    def prepare_dataset_from_indices(self, indices):
        """
        Create a TensorDataset from specific indices of this dataset
        
        Parameters:
        -----------
        indices: list of indices to include
        
        Returns:
        --------
        torch.utils.data.TensorDataset: Dataset containing features and labels
    """
        features = []
        targets = []
        
        for idx in indices:
            feature_tensor, label = self[idx]
            features.append(feature_tensor)
            targets.append(label)
        
        # Create dataset
        return torch.utils.data.TensorDataset(
            torch.stack(features),
            torch.tensor(targets)
        )

    def prepare_dataset_from_files(self, audio_files, labels):
        """
        Create a dataset from specified audio files and labels
        
        Parameters:
        -----------
        audio_files: list of paths to audio files
        labels: list of labels corresponding to audio files
        
        Returns:
        --------
        torch.utils.data.TensorDataset: Dataset containing features and labels
        """
        features = []
        targets = []
        
        for audio_path, label in zip(audio_files, labels):
            if self.extract_features:
                # Extract features
                feature_image = self._prepare_features(audio_path)
                
                # Convert to PyTorch tensor with channels first
                feature_tensor = torch.tensor(feature_image).float()
                feature_tensor = feature_tensor.permute(2, 0, 1)  # [channels, height, width]
            else:
                # Load waveform
                waveform, _ = torchaudio.load(audio_path)
                if self.transform:
                    feature_tensor = self.transform(waveform)
                else:
                    feature_tensor = waveform
            
            features.append(feature_tensor)
            targets.append(label)
        
        # Create dataset
        return torch.utils.data.TensorDataset(
            torch.stack(features),
            torch.tensor(targets)
        )

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

    def _prepare_features(self, audio_path):
        """
        Load audio file and extract features
        :return: processed feature image with shape [height, width, channels]
        """
        y, sr = librosa.load(audio_path, sr=self.sample_rate)

        # Short-time fourier transform
        stft = np.abs(librosa.stft(y, n_fft=255, hop_length=512))
        stft_padded = self._padding(stft, 128, self.feature_size)

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_fft=255, hop_length=512, n_mfcc=128)
        mfccs_padded = self._padding(mfccs, 128, self.feature_size)
        
        # spectral centroid
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spec_centroid_norm = self._normalize(spec_centroid)
        
        # chroma features
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_stft_norm = self._normalize(chroma_stft)
        
        # spectral bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spec_bw_norm = self._normalize(spec_bw)

        # build image layers
        image = np.array([self._padding(spec_bw_norm.reshape(1, -1), 1, self.feature_size)]).reshape(1, self.feature_size)
        image = np.append(image, self._padding(spec_centroid_norm.reshape(1, -1), 1, self.feature_size), axis=0)

        # repeat padded features
        for i in range(0, 9):
            image = np.append(image, self._padding(spec_bw_norm.reshape(1, -1), 1, self.feature_size), axis=0)
            image = np.append(image, self._padding(spec_centroid_norm.reshape(1, -1), 1, self.feature_size), axis=0)
            image = np.append(image, self._padding(chroma_stft_norm, 12, self.feature_size), axis=0)
        
        # Stack features as channels
        image = np.dstack((image, stft_padded))
        image = np.dstack((image, mfccs_padded))

        return image