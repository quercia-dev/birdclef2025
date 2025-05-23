from efficient_b_train import *
from train_utils import *

import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

# --- Hardcoded paths ---
MODEL_PATH = "model/20250521_124614_efficientb0/model_fold0.pth"
SPECTROGRAM_NPY = "data/birdclef2025_melspec_5sec_256_256_soundscapes.npy"
DATASET_CSV = "data/train_soundscapes.csv"
TAXONOMY_CSV = "data/taxonomy.csv"
OUTPUT_CSV = "efficient_b_labelling_soundscape.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

cfg: CFG = checkpoint['cfg']

taxonomy_df = pd.read_csv(TAXONOMY_CSV)
class_names = taxonomy_df['primary_label'].tolist()

model = BirdCLEFModel(cfg).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load spectrograms and metadata
spectrograms = np.load(SPECTROGRAM_NPY, allow_pickle=True).item()
df = pd.read_csv(DATASET_CSV)

# Create dataset and dataloader
dataset = BirdCLEFDatasetFromNPY(df, cfg=cfg, spectrograms=spectrograms)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

# Predict
results = []
with torch.no_grad():
    for batch in tqdm(loader, desc="Predicting"):
        inputs = batch["melspec"].to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        ids = batch["id"]
        results.extend(zip(ids, [class_names[p] for p in preds]))

# Save predictions
result_df = pd.DataFrame(results, columns=["samplename", "prediction"])
result_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved predictions to {OUTPUT_CSV}")