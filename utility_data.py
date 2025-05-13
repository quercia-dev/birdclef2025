import os
import pandas as pd
import torch
import sys
import subprocess
import torchaudio
import librosa
import numpy as np
from typing import Dict, Optional, Tuple
from mutagen.oggvorbis import OggVorbis
from concurrent.futures import ThreadPoolExecutor
import math
import soundfile
import ast


def get_audio_metadata(audio_path: str) -> Dict:
    """Extract metadata from audio file"""
    file_size = os.path.getsize(audio_path)  # in bytes
    try:
        audio = OggVorbis(audio_path)

        # Metadata extraction
        duration = audio.info.length  # in seconds
        sample_rate = audio.info.sample_rate
        channels = audio.info.channels
        bitrate = audio.info.bitrate // 1000  # in kbps

        return {
            "duration": duration,
            "sample_rate": sample_rate,
            "channels": channels,
            "bitrate": bitrate,
            "file_size": file_size
        }
    except Exception as e:
        print(f"Error loading metadata for {audio_path}: {e}")
        return {
            "duration": 0,
            "sample_rate": 0,
            "channels": 0,
            "bitrate": 0,
            "file_size": 0
        }


def crop_and_save(segment: int, input_folder:str, output_folder:str, filepath:str, transform=None, delete_source:bool=False) -> list:
    """
    Load every file indicated by the ta_metadata file, 
    split into segments, and save each segment as a separate file.
    The last segment is padded with zeroes if needed.
    Returns a list of filepaths of exported files
    """
    
    try:
        input_path = os.path.join(input_folder, filepath)
        
        sig, sample_rate = torchaudio.load(input_path)
        total_len = sig.shape[1]
        num_segments = math.ceil(total_len / segment)

        dir_name = os.path.dirname(filepath)
        base_name, _ = os.path.basename(filepath).rsplit('.', 1)
        os.makedirs(os.path.join(output_folder, dir_name), exist_ok=True)

        filenames = []
        for i in range(num_segments):
            start = i * segment
            end = start + segment

            # If this is the last segment
            if end >= total_len:
                remaining = total_len - start

                # If the remaining is less than half a segment,
                # shift window to capture the last `segment` samples
                if remaining > segment // 2:
                    start = max(0, total_len - segment)
                    end = total_len

            segment_data = sig[:, start:end]

            # If needed (very rare), pad short segment to full length
            if segment_data.shape[1] < segment:
                pad_amount = segment - segment_data.shape[1]
                segment_data = torch.cat([segment_data, torch.zeros(1, pad_amount)], dim=1)

            isOGG = transform is None
            
            if isOGG:
                extension = 'ogg'            
                segment_filename = os.path.join(dir_name, f'{base_name}_{i}.{extension}')
                segment_path = os.path.join(output_folder, segment_filename)                
                torchaudio.save(segment_path, segment_data, sample_rate)
                
            else:
                extension = 'pt'
                segment_filename = os.path.join(dir_name, f'{base_name}_{i}.{extension}')
                segment_path = os.path.join(output_folder, segment_filename)            
                segment_data = transform(segment_data)
                torch.save(segment_data, segment_path)
            
            filenames += [segment_filename]

        if delete_source:
            os.remove(input_path)
        
        return filenames

    except Exception as e:
        print(f'Error processing {filepath}: {e}')
        return []

DEFAULT_SAMPLE_RATE = 32000
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = 512
DEFAULT_N_MFCC = 64
DEFAULT_N_MELS = 64
DEFAULT_FEATURE_SIZE = 512

class AudioDataset(torch.utils.data.Dataset):
    """Dataset class for audio processing with feature extraction capabilities."""

    def _process_row(self, index_row_tuple):
        """
        Wrapper for `calculate_label_tensor`, making it iterable
        """
        idx, row = index_row_tuple
        return idx, self.calculate_label_tensor(row)

    def _get_audio_params(self, audio_params: Optional[Dict] = None) -> Tuple[int, int, int, int, int, int]:
        params = audio_params or {}

        sample_rate = params.get("sample_rate", DEFAULT_SAMPLE_RATE)
        n_fft = params.get("n_fft", DEFAULT_N_FFT)
        hop_length = params.get("hop_length", DEFAULT_HOP_LENGTH)
        n_mfcc = params.get("n_mfcc", DEFAULT_N_MFCC)
        n_mels = params.get("n_mels", DEFAULT_N_MELS)
        feature_size = params.get("feature_size", DEFAULT_FEATURE_SIZE)
        
        return sample_rate, n_fft, hop_length, n_mfcc, n_mels, feature_size
    
    def _get_audio_transform(self, feature_mode:str):
        if feature_mode == 'mel':
                mel = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels,
                    f_min=40,
                    f_max=15000,
                    power=2.0,
                )
                return mel

        elif feature_mode == 'mfcc':
            mfcc = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=self.n_mfcc,
                melkwargs={
                    'n_fft': self.n_fft,
                    'hop_length': self.hop_length,
                    'n_mels': self.n_mels,
                    'f_min': 40,
                    'f_max': 15000,
                    'power': 2.0,
                }
            )
            return mfcc
        else:
            return None
        
    def _get_audio_data(self, csv_path: str) -> pd.DataFrame:
        if os.path.exists(csv_path):
            data = pd.read_csv(csv_path)

            # parses to list the columns                
            for col in ['type', 'secondary_labels', 'filename']:
                if col in data.columns:
                    data[col] = data[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x)

            # create a new row for each filename
            return data.explode('filename', ignore_index=True)
        else:
            return pd.DataFrame()
        
    def _get_torch_labels(self) -> Dict:
        if ('primary_label' in self.data.columns) and ('secondary_labels' in self.data.columns):
            with ThreadPoolExecutor() as executor:
                results = executor.map(self._process_row, self.data.iterrows())
                return dict(results)
        
        
    def __init__(
        self, 
        datafolder:str="data",
        metadata_csv: str="train.csv",
        audio_dir: str="train_audio",
        transform=None, 
        metadata: bool=False, 
        feature_mode: str='mel', 
        m:float=0.65,
        audio_params: Optional[Dict]=None):
        """
        datafolder: name of the folder containing the data
        metadata_csv: name of the csv metadata file
        audio_dir: path to 'train_audio/'
        transform: torch audio transform the waveform. 
            If unspecified, the feature_mode and audio_params are used to construct one.
        m: probability weight to give the primary label 
        feature_mode: method of feature extraction when loading the file ('' - raw, 'mel', 'mfcc')
        audio_params: parameters for audio feature extraction
        """
        self.datafolder = os.path.join(".", datafolder, "")
        self.audio_dir = os.path.join(self.datafolder, audio_dir) 
                
        self.feature_mode = feature_mode

        (self.sample_rate, 
        self.n_fft, 
        self.hop_length, 
        self.n_mfcc, 
        self.n_mels, 
        self.feature_size) = self._get_audio_params(audio_params)
            
        self.transform = transform
            
        if transform is None:
            self.transform = self._get_audio_transform(feature_mode)
                            
        # 5 sec is hardcoded. This is because the final 
        # classification task provides 5 sec audios
        self.segment = 5*self.sample_rate

        if metadata_csv:
            csv_path = os.path.join(self.datafolder, metadata_csv)
            self.data = self._get_audio_data(csv_path)
        else:
            self.data = self._load_audio_data(self.audio_dir)

        if metadata and not self.data.empty:
            self._extract_metadata()

        # sort by alphabetical order, then map species name to label index
        self.classes = ['null'] + sorted(self.data["primary_label"].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        label_durations = self.data.groupby('primary_label')['duration'].sum()
        label_proportions = label_durations / self.data['duration'].sum()
        self.alphas = torch.tensor(1 / label_proportions.values, dtype=torch.float32)
        
        # sets the primary label probability weight
        if m < 0 or m > 1:
            m = 0.5
        self.m = m
        
        self.labels = self._get_torch_labels()
        
    def filter_by_values(self, column: str, selected_labels: list, include:bool=True) -> bool:
        """Filter the DataFrame: when include is True, only includes rows where the column's value is in selected_labels.
        Otherwise, if include == False, excludes all selected_labels
        Returns True if the column exists, otherwise does nothing and returns False."""
        colExists = column in self.data.columns
        if colExists:
            if include:        
                self.data = self.data[self.data[column].isin(selected_labels)]                
            else:
                self.data = self.data[~self.data[column].isin(selected_labels)]                                
        return colExists
                        
    def calculate_label_tensor(self, row: pd.Series) -> torch.Tensor:
        """
        Generates a probabilistic label tensor from a row with primary and secondary labels.  
        Primary gets weight `m`; secondary labels share the remaining `(1 - m)` mass.  
        Returns a tensor of shape [num_classes] with class probabilities.
        """
        num_classes = len(self.class_to_idx)
        label_tensor = torch.zeros(num_classes)

        primary_label = row['primary_label']
        primary_idx = self.class_to_idx.get(primary_label, 0)
        label_tensor[primary_idx] = self.m

        secondary_labels = row.get('secondary_labels', [])
        if secondary_labels:
            remaining_prob = (1 - self.m) / len(secondary_labels)
            for sec_label in secondary_labels:
                sec_idx = self.class_to_idx.get(sec_label, 0)
                label_tensor[sec_idx] += remaining_prob
        else: # adds the remaining probability mass to the other labels equally
            remaining_prob = (1 - self.m) / (num_classes - 1)
            label_tensor += remaining_prob
            label_tensor[primary_idx] -= remaining_prob  # undo addition at primary index

        return label_tensor
        
    def preprocess(self, output: str='train_proc', delete_source:bool=False):
        """
        Pre-processes the data using a transform.
        Saves the new table to the datafolder and
        the new audio files to the 'datafolder / output'
        Returns a new AudioDataset file.
        """
        
        if output == 'train':
            raise FileExistsError("I can't let you overwrite 'train.csv', Hal")
        
        output_path = os.path.join(self.datafolder, output)
        
        data = self.data.copy()
        data.loc[:, 'idx'] = data.index
        # apply crop_and_save to each row and set the filename column as a list value 
        data.loc[:, 'filename'] = data.apply(lambda row : crop_and_save(self.segment, self.audio_dir, output_path, row['filename'], self.transform, delete_source), axis=1)
        
        data.to_csv(os.path.join(self.datafolder, f'{output}.csv'), index=False)
        

    def _extract_metadata(self) -> None:
        """Extract audio metadata and add to dataframe."""
        metadata_df = self.data["filename"].apply(lambda filename: get_audio_metadata(os.path.join(self.audio_dir, filename)))
        # unpack dictionary and assign to new columns
        metadata_df = pd.DataFrame(metadata_df.tolist())
        self.data = pd.concat([self.data, metadata_df], axis=1)
       
            
    def __len__(self):
        return len(self.data)

    def get_feature(self, idx) -> torch.Tensor:
        row = self.data.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row["filename"])

        if self.feature_mode == '':
            return torch.load(audio_path, weights_only=True)
        elif self.feature_mode == 'mel':
            mel_spec, _ = self._extract_spectrogram(audio_path)
            return mel_spec.clone().detach()
        elif self.feature_mode == 'rich':
            return self._get_features(audio_path)
        else:
            return self._get_waveform(audio_path)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_feature(idx), self.labels[idx]
       
        
    def get(self, idx: int):
        # get metadata row of specified index
        row = self.data.iloc[idx]

        audio_path = os.path.join(self.audio_dir, row["filename"])
        try:
            waveform, _ = torchaudio.load(audio_path)
        except Exception:
            print(f"Error loading {audio_path}")
            return torch.zeros(1, 16000), -1 # dummy data if missing file

        return waveform, self.class_to_idx[row["primary_label"]]
    

    def _extract_spectrogram(self, audio_path: str) -> Tuple[torch.Tensor, bool]:
        """
        Helper method to extract features using the self.transform (mel or mfcc).
        """
        if self.transform is None:
            raise ValueError("Feature transform is not set. Make sure to initialize 'self.transform' before calling this method.")

        try:
            waveform, _ = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if self.sample_rate != 32000:
                resampler = torchaudio.transforms.Resample(orig_freq=_, new_freq=self.sample_rate)
                waveform = resampler(waveform)

            features = self.transform(waveform)  # Applies mel or mfcc transform
            return features, True

        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            dummy_shape = (self.n_mfcc if isinstance(self.transform, torchaudio.transforms.MFCC) else self.n_mels, 1000)
            return torch.zeros(dummy_shape, dtype=torch.float32), False


    def _get_features(self, audio_path: str) -> torch.Tensor:
        """Extract audio features for CNN."""
        try:
            features = self.generate_features(audio_path)
            features_tensor = torch.from_numpy(features).clone().detach()
            features_tensor = features_tensor.permute(2, 0, 1)  # [channels, height, width]
            return features_tensor
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return torch.zeros(3, 64, self.feature_size)
  
    
    def _get_waveform(self, audio_path: str) -> torch.Tensor:
        """Load audio waveform."""
        try:
            waveform, _ = torchaudio.load(audio_path)
            if self.transform:
                waveform = self.transform(waveform)
            return waveform
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(1, 16000)


    def open(self, idx: int) -> None:
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


    def locate(self, idx: int) -> None:
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


    def get_features(self, idx: int) -> Tuple[np.ndarray, int]:
        """Extract and return features for a specific sample"""
        row = self.data.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row["filename"])
        features = self.generate_features(audio_path)
        label = self.labels[idx]
        return features, label


    def _get_path_at_index(self, idx: int) -> str:
        """Get the full path to the audio file at the given index."""
        row = self.data.iloc[idx]
        return os.path.join(self.audio_dir, row["filename"])
        
        
    def _load_audio_data(self, audio_dir: str) -> pd.DataFrame:
        """Constructs a DataFrame from the audio files in the audio_dir path.

        - Supports .ogg, .wav, .mp3 files.
        - Adds 'primary_label' column for .ogg files.
        """
        data_rows = []

        if not os.path.exists(audio_dir):
            print(f"Warning: Audio directory does not exist: {audio_dir}")
            return pd.DataFrame(columns=["filename", "primary_label"])

        for root, _, files in os.walk(audio_dir):
            for file in files:
                if file.lower().endswith(('.ogg', '.wav', '.mp3')):
                    rel_path = os.path.relpath(os.path.join(root, file), audio_dir)
                    primary_label = os.path.splitext(file)[0] if file.lower().endswith('.ogg') else None
                    data_rows.append({
                        "filename": rel_path,
                        "primary_label": primary_label
                    })

        return pd.DataFrame(data_rows)


    def _normalize(self, array: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1]"""
        min_val = np.min(array)
        max_val = np.max(array)
        if max_val > min_val:
            return (array - min_val) / (max_val - min_val)
        return np.zeros_like(array)


    def _padding(self, array: np.ndarray, height: int, width: int) -> np.ndarray:
        """
        Pad or trim array to specified dimensions
        :param array: numpy array
        :param xx: desired height
        :param yy: desired width
        :return: padded or trimmed array
        """
        h, w = array.shape
        
        # If array is already equal to or wider than target, center-crop
        if w >= width:
            start_idx = (w - width) // 2
            result = array[:, start_idx:start_idx+width]
            
            # If height needs adjustment
            if h != height:
                # Create new array with target dimensions
                padded = np.zeros((height, width))
                # Determine how much to use from original
                use_h = min(h, height)
                # Center the content vertically
                start_h = (height - use_h) // 2
                # Copy content
                padded[start_h:start_h+use_h, :] = result[:use_h, :]
                return padded
            return result
        
        # If array needs padding
        padded = np.zeros((height, width))
        
        # Center the content both vertically and horizontally
        start_h = (height - h) // 2 if h < height else 0
        start_w = (width - w) // 2 if w < width else 0
        
        # Copy data, handling both dimensions
        use_h = min(h, height)
        use_w = min(w, width)
        
        padded[start_h:start_h+use_h, start_w:start_w+use_w] = array[:use_h, :use_w]
        return padded


    def generate_features(self, audio_path: str) -> np.ndarray:
        try:
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                return np.zeros((64, self.feature_size, 3))
            
            sig, sr = torchaudio.load(audio_path)  # sig shape: [channels, time]
            
            if sig.shape[0] > 1:
                sig = sig.mean(dim=0, keepdim=True)  # shape: [1, time]

            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                sig = resampler(sig)

            y_cut = sig.squeeze().numpy()

            if np.all(y_cut == 0) or len(y_cut) == 0:
                print(f"Warning: Audio file {audio_path} loaded as silence or empty")
                return np.zeros((64, self.feature_size, 3))

            features = {
                'mel_spec': self._compute_mel_spectrogram(y_cut),
                'mfccs': self._compute_mfccs(y_cut),
                'spectral': self._compute_spectral_features(y_cut),
            }

            # Stack them along last axis
            result = np.stack([
                features['spectral'],
                features['mel_spec'],
                features['mfccs']
            ], axis=2)
            
            return result
            
        except Exception as e:
            print(f"Error generating features for {audio_path}: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((64, self.feature_size, 3))        
        
        
    def _compute_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Compute and normalize mel spectrogram."""
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=self.sample_rate, n_fft=self.n_fft, 
            hop_length=self.hop_length, n_mels=self.n_mels,
            fmax=self.sample_rate/2 
        )
        mel_spec = self._normalize(np.abs(mel_spec))
        return self._padding(mel_spec, 64, self.feature_size)
    
    
    def _compute_mfccs(self, y: np.ndarray) -> np.ndarray:
        """Compute and normalize MFCCs."""
        mfccs = librosa.feature.mfcc(
            y=y, sr=self.sample_rate, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mfcc=self.n_mfcc
        )
        mfccs = self._normalize(mfccs)
        return self._padding(mfccs, 64, self.feature_size)
    
    
    def _compute_spectral_features(self, y: np.ndarray) -> np.ndarray:
        """Compute and combine various spectral features into a single channel."""
        # Initialize third channel
        channel3 = np.zeros((64, self.feature_size))
        
        # Spectral centroid (1 row)
        spec_centroid = librosa.feature.spectral_centroid(
            y=y, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        spec_centroid = self._padding(self._normalize(spec_centroid), 1, self.feature_size)
        
        # Spectral bandwidth (1 row)
        spec_bw = librosa.feature.spectral_bandwidth(
            y=y, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        spec_bw = self._padding(self._normalize(spec_bw), 1, self.feature_size)
        
        # Chroma STFT (12 rows)
        chroma_stft = librosa.feature.chroma_stft(
            y=y, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        chroma_stft = self._padding(self._normalize(chroma_stft), 12, self.feature_size)
        
        # Add spectral bandwidth and centroid
        channel3[0:1] = spec_bw
        channel3[1:2] = spec_centroid
        
        # Add chroma (12 rows)
        channel3[2:14] = chroma_stft
        
        # Fill remaining rows
        current_row = 14
        while current_row < 64:
            if current_row % 2 == 0 and current_row + 1 <= 64:
                channel3[current_row:current_row+1] = spec_bw
            elif current_row + 1 <= 64:
                channel3[current_row:current_row+1] = spec_centroid
            current_row += 1
        
        return channel3
    
    
if __name__ == '__main__':
    audio_params = {
        'sample_rate': 32000,
        'n_fft': 1024,
        'hop_length': 501,
        'n_mfcc': 128,
        'n_mels': 128,
        'feature_size': 2048
    }

    AudioDataset(
        datafolder="data",
        metadata_csv="train.csv",
        audio_dir="train_audio",
        feature_mode="",
        audio_params=audio_params,
        metadata=True
    ).preprocess(output="train_proc")
    
    # split the soundscapes as well
    
    AudioDataset(
        datafolder="data",
        metadata_csv="",
        audio_dir="train_soundscapes",
        feature_mode="",
        audio_params=audio_params,
        metadata=True
    ).preprocess(output="train_soundscapes_proc")