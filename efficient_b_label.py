from efficient_b_train import *
from train_utils import *

import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

# --- CFG ---
class CFG:
    seed = 42
    debug = False 
    apex = False
    print_freq = 100
    num_workers = 2

    OUTPUT_DIR = 'model/'

    train_datadir = 'data/train_audio'
    train_csv = 'data/train.csv'
    taxonomy_csv = 'data/taxonomy.csv'

    spectrogram_npy = 'data/birdclef2025_melspec_5sec_256_256_soundscapes.npy'

    model_name = 'efficientnet_b0'  
    pretrained = True
    in_channels = 1

    LOAD_DATA = True  
    FS = 32000
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (256, 256)

    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10  
    batch_size = 32  
    criterion = 'BCEWithLogitsLoss'

    n_fold = 5
    selected_folds = [0, 1, 2, 3, 4]   

    optimizer = 'AdamW'
    lr = 5e-4 
    weight_decay = 1e-5

    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = epochs

    aug_prob = 0.5  
    mixup_alpha = 0.5  

    def update_debug_settings(self):
        if self.debug:
            self.epochs = 1
            self.selected_folds = [0]


# --- Hardcoded paths ---
def main():
    MODEL_PATH = "model/20250519_182132_efficientb0/model_fold0.pth"
    SPECTROGRAM_NPY = "data/birdclef2025_melspec_5sec_256_256_soundscapes.npy"
    DATASET_CSV = "data/train_soundscapes.csv"
    TAXONOMY_CSV = "data/taxonomy.csv"
    OUTPUT_CSV = "efficient_b_labelling_soundscape.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        cfg = checkpoint['cfg']
    except KeyError:
        # If cfg is not in checkpoint, use default CFG
        print("Warning: 'cfg' not found in checkpoint, using default CFG")
        cfg = CFG()
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Using default CFG")
        cfg = CFG()

    # Load taxonomy
    taxonomy_df = pd.read_csv(TAXONOMY_CSV)
    class_names = taxonomy_df['primary_label'].tolist()

    try:
        # Initialize model
        model = BirdCLEFModel(cfg).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load spectrograms and metadata
    try:
        spectrograms = np.load(SPECTROGRAM_NPY, allow_pickle=True).item()
        df = pd.read_csv(DATASET_CSV)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create dataset and dataloader with num_workers=0 to avoid multiprocessing issues
    dataset = BirdCLEFDatasetFromNPY(df, cfg=cfg, spectrograms=spectrograms)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Predict
    results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            try:
                inputs = batch["melspec"].to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                ids = batch["id"]
                results.extend(zip(ids, [class_names[p] for p in preds]))
            except Exception as e:
                print(f"Error during prediction: {e}")
                continue
    
    # Save predictions
    if results:
        result_df = pd.DataFrame(results, columns=["samplename", "prediction"])
        result_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved predictions to {OUTPUT_CSV}")
    else:
        print("No predictions were made")

if __name__ == "__main__":
    main()