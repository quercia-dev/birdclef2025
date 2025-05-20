from train_utils import *
from utility_plots import plot_confusion_matrix

import os
import logging
import random
import gc
import time
import cv2
import math
import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm.auto import tqdm

import timm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


class CFG:

    seed = 42
    debug = False 
    apex = False
    print_freq = 100
    num_workers = 2

    OUTPUT_DIR = 'model/'

    train_datadir = 'data/train_audio'
    train_csv = 'data/train.csv'
    #test_soundscapes = 'data/test_soundscapes'
    #submission_csv = 'data/sample_submission.csv'
    taxonomy_csv = 'data/taxonomy.csv'

    spectrogram_npy = 'data/birdclef2025_melspec_5sec_256_256.npy'

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

cfg = CFG()


set_seed(cfg.seed)


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
        cfg.num_classes = len(taxonomy_df)

        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            drop_rate=0.2,
            drop_path_rate=0.2
        )

        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.feat_dim = backbone_out

        self.classifier = nn.Linear(backbone_out, cfg.num_classes)

        self.mixup_enabled = hasattr(cfg, 'mixup_alpha') and cfg.mixup_alpha > 0
        if self.mixup_enabled:
            self.mixup_alpha = cfg.mixup_alpha

    def forward(self, x, targets=None):

        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None

        features = self.backbone(x)

        if isinstance(features, dict):
            features = features['features']

        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)

        logits = self.classifier(features)

        if self.training and self.mixup_enabled and targets is not None:
            loss = self.mixup_criterion(F.binary_cross_entropy_with_logits, 
                                       logits, targets_a, targets_b, lam)
            return logits, loss

        return logits

    def mixup_data(self, x, targets):
        """Applies mixup to the data batch"""
        batch_size = x.size(0)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        indices = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[indices]

        return mixed_x, targets, targets[indices], lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Applies mixup to the loss function"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):

    model.train()
    losses = []
    all_targets = []
    all_outputs = []

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")

    for step, batch in pbar:

        if isinstance(batch['melspec'], list):
            batch_outputs = []
            batch_losses = []

            for i in range(len(batch['melspec'])):
                inputs = batch['melspec'][i].unsqueeze(0).to(device)
                target = batch['target'][i].unsqueeze(0).to(device)

                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, target)
                loss.backward()

                batch_outputs.append(output.detach().cpu())
                batch_losses.append(loss.item())

            optimizer.step()
            outputs = torch.cat(batch_outputs, dim=0).numpy()
            loss = np.mean(batch_losses)
            targets = batch['target'].numpy()

        else:
            inputs = batch['melspec'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if isinstance(outputs, tuple):
                outputs, loss = outputs  
            else:
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()

        all_outputs.append(outputs)
        all_targets.append(targets)
        losses.append(loss if isinstance(loss, float) else loss.item())

        pbar.set_postfix({
            'train_loss': np.mean(losses[-10:]) if losses else 0,
            'lr': optimizer.param_groups[0]['lr']
        })

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)

    return avg_loss, auc


def validate(model, loader, criterion, device):

    model.eval()
    losses = []
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            if isinstance(batch['melspec'], list):
                batch_outputs = []
                batch_losses = []

                for i in range(len(batch['melspec'])):
                    inputs = batch['melspec'][i].unsqueeze(0).to(device)
                    target = batch['target'][i].unsqueeze(0).to(device)

                    output = model(inputs)
                    loss = criterion(output, target)

                    batch_outputs.append(output.detach().cpu())
                    batch_losses.append(loss.item())

                outputs = torch.cat(batch_outputs, dim=0).numpy()
                loss = np.mean(batch_losses)
                targets = batch['target'].numpy()

            else:
                inputs = batch['melspec'].to(device)
                targets = batch['target'].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                outputs = outputs.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()

            all_outputs.append(outputs)
            all_targets.append(targets)
            losses.append(loss if isinstance(loss, float) else loss.item())

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)

    predicted_classes = np.argmax(all_outputs, axis=1)

    if all_targets.ndim == 2 and all_targets.shape[1] > 1:
        all_targets = np.argmax(all_targets, axis=1)
    elif all_targets.ndim == 1:
        pass  # already correct shape
    else:
        raise ValueError(f"Unexpected shape for all_targets: {all_targets.shape}")

    val_acc = accuracy_score(all_targets, predicted_classes)
    val_acc_balanced = balanced_accuracy_score(all_targets, predicted_classes)
    return avg_loss, auc, val_acc, val_acc_balanced

def run_training(df, cfg):
    """Training function that can either use pre-computed spectrograms or generate them on-the-fly"""

    results_folder = f'./model/{time.strftime("%Y%m%d_%H%M%S")}_efficientb0'
    os.makedirs(results_folder, exist_ok=True)

    metrics_file = os.path.join(results_folder, 'metrics.csv')
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'fold', 'epoch', 'train_loss', 'train_auc', 'val_loss', 
                         'val_auc', 'val_acc', 'val_bal_acc', 'lr', 'time'])

    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    species_ids = taxonomy_df['primary_label'].tolist()
    cfg.num_classes = len(species_ids)

    if cfg.debug:
        cfg.update_debug_settings()

    # df Subset check, needed after random_split 90-10 split
    if hasattr(df, 'indices') and hasattr(df, 'dataset'):
        # It's a Subset from random_split, convert back to DataFrame
        train_df = df.dataset.iloc[df.indices].reset_index(drop=True)
    else:
        train_df = df  # Already a DataFrame

    spectrograms = None
    if cfg.LOAD_DATA:
        print("Loading pre-computed mel spectrograms from NPY file...")
        try:
            spectrograms = np.load(cfg.spectrogram_npy, allow_pickle=True).item()
            print(f"Loaded {len(spectrograms)} pre-computed mel spectrograms")
        except Exception as e:
            print(f"Error loading pre-computed spectrograms: {e}")
            print("Will generate spectrograms on-the-fly instead.")
            cfg.LOAD_DATA = False

    if not cfg.LOAD_DATA:
        print("Will generate spectrograms on-the-fly during training.")
        if 'filepath' not in df.columns:
            df['filepath'] = cfg.train_datadir + '/' + df.filename
        if 'samplename' not in df.columns:
            df['samplename'] = df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

    skf = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)

    best_scores = []
    best_fold_indices = []
    
    global_step = 0
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['primary_label'])):
        if fold not in cfg.selected_folds:
            continue
        epoch_start_time = time.time()
        
        print(f'\n{"="*30} Fold {fold} {"="*30}')

        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

        print(f'Training set: {len(fold_train_df)} samples')
        print(f'Validation set: {len(fold_val_df)} samples')

        train_dataset = BirdCLEFDatasetFromNPY(fold_train_df, cfg, spectrograms=spectrograms, mode='train')
        val_dataset = BirdCLEFDatasetFromNPY(fold_val_df, cfg, spectrograms=spectrograms, mode='valid')

        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=True, 
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=False, 
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

        model = BirdCLEFModel(cfg).to(cfg.device)
        optimizer = get_optimizer(model, cfg)
        criterion = get_criterion(cfg)

        if cfg.scheduler == 'OneCycleLR':
            scheduler = lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.lr,
                steps_per_epoch=len(train_loader),
                epochs=cfg.epochs,
                pct_start=0.1
            )
        else:
            scheduler = get_scheduler(optimizer, cfg)

        best_auc = 0
        best_epoch = 0

        for epoch in range(cfg.epochs):
            epoch_start_time = time.time()
            
            print(f"\nEpoch {epoch+1}/{cfg.epochs}")

            train_loss, train_auc = train_one_epoch(
                model, 
                train_loader, 
                optimizer, 
                criterion, 
                cfg.device,
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None
            )

            val_loss, val_auc, val_acc, val_acc_bal = validate(model, val_loader, criterion, cfg.device)

            if scheduler is not None and not isinstance(scheduler, lr_scheduler.OneCycleLR):
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}, Val Bal Acc: {val_acc_bal:.4f}")

            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - epoch_start_time
            
            with open(metrics_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    global_step, fold, epoch + 1,
                    f"{train_loss:.6f}", f"{train_auc:.6f}",
                    f"{val_loss:.6f}", f"{val_auc:.6f}",
                    f"{val_acc:.6f}", f"{val_acc_bal:.6f}",
                    f"{lr:.8f}", f"{elapsed:.2f}"
                ])

            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch + 1
                print(f"New best AUC: {best_auc:.4f} at epoch {best_epoch}")

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'epoch': epoch,
                    'val_auc': val_auc,
                    'train_auc': train_auc,
                    'cfg': cfg
                }, os.path.join(results_folder, f"model_fold{fold}.pth"))
                
            global_step += 1

        best_scores.append(best_auc)
        best_fold_indices.append(fold)
        print(f"\nBest AUC for fold {fold}: {best_auc:.4f} at epoch {best_epoch}")

        # Clear memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    best_score_idx = np.argmax(best_scores)
    best_fold = best_fold_indices[best_score_idx]
    best_model_path = os.path.join(results_folder, f"model_fold{best_fold}.pth")

    print("\n" + "="*60)
    print("Cross-Validation Results:")
    for fold, score in enumerate(best_scores):
        print(f"Fold {cfg.selected_folds[fold]}: {score:.4f}")
    print(f"Mean AUC: {np.mean(best_scores):.4f}")
    print("="*60)

    return results_folder, best_model_path

if __name__ == "__main__":

    print("\nLoading training data...")
    train_df = pd.read_csv(cfg.train_csv)
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    class_names = taxonomy_df['primary_label'].tolist()

    print("\nStarting training...")
    print(f"LOAD_DATA is set to {cfg.LOAD_DATA}")
    if cfg.LOAD_DATA:
        print("Using pre-computed mel spectrograms from NPY file")
    else:
        print("Will generate spectrograms on-the-fly during training")


    train_len = int(0.9 * len(train_df))
    val_len = len(train_df) - train_len

    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(train_df, [train_len, val_len], generator=generator)
    
    # Store spectrograms reference before training
    if cfg.LOAD_DATA:
        try:
            spectrograms = np.load(cfg.spectrogram_npy, allow_pickle=True).item()
        except Exception as e:
            spectrograms = None
            print(f"Error loading pre-computed spectrograms: {e}")
    else:
        spectrograms = None
    
    results_folder, best_model_path = run_training(train_set, cfg)

    print("\nTraining complete!")

    print("Unseen validation")

    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, weights_only = False)
        cfg = checkpoint['cfg']  # Load the config used during training
        model = BirdCLEFModel(cfg).to(cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set the model to evaluation mode
    else:
        raise FileNotFoundError(f"No saved model found at {best_model_path}")

    # Properly extract the validation dataframe
    val_indices = val_set.indices
    unseen_val_df = train_df.iloc[val_indices].reset_index(drop=True)

    # Create dataset and dataloader for unseen validation
    unseen_val_dataset = BirdCLEFDatasetFromNPY(unseen_val_df, cfg, spectrograms=spectrograms, mode='valid')
    
    # Import required for collate_fn if not already defined
    from sklearn.metrics import confusion_matrix
    
    unseen_val_loader = DataLoader(
        unseen_val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Run validation on the unseen data
    criterion = get_criterion(cfg)
    val_loss, val_auc, val_acc, val_acc_bal = validate(model, unseen_val_loader, criterion, cfg.device)

    print(f"Unseen Validation Loss: {val_loss:.4f}, Unseen Validation AUC: {val_auc:.4f}, Unseen Validation Acc: {val_acc:.4f}, Unseen Validation Bal Acc: {val_acc_bal:.4f}")

    # Generate predictions for confusion matrix
    all_targets = []
    all_predicted = []

    with torch.no_grad():
        for batch in tqdm(unseen_val_loader, desc="Unseen Validation Prediction"):
            inputs = batch['melspec'].to(cfg.device)
            targets = batch['target'].to(cfg.device)

            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = torch.argmax(targets, dim=1).cpu().numpy()

            all_targets.extend(targets)
            all_predicted.extend(predicted)

    confusion_file = os.path.join(results_folder, "confusion_results.csv")

    # Write header if file doesn't exist
    if not os.path.exists(confusion_file):
        with open(confusion_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['targets', 'predictions'])
    
    # Append results
    with open(confusion_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            str(all_targets), str(all_predicted),
        ])

    # Create confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_predicted)

    plot_path = os.path.join(results_folder, "final_confusion_matrix.png")

    plot_confusion_matrix(
    cm=confusion_matrix, 
    classes=class_names, 
    normalize=False,
    title='Confusion Matrix',
    save_path=plot_path
    )


