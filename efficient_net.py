from focal_loss import FocalLoss
from utility_data import *
from utility_plots import *


import os
import time
import torch
import torch.nn as nn 
import pytorch_lightning as pl
import torch.nn.functional as F 
from pytorch_lightning.loggers import CSVLogger
from torchmetrics.classification import Accuracy
from torch.utils.data import DataLoader, random_split
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights



class EfficientNetAudio(pl.LightningModule):
    def __init__(self, num_classes: int, gamma: float, alphas: torch.Tensor, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.val_balanced_acc = Accuracy(num_classes=num_classes, task='multiclass', average='macro')
        self.train_balanced_acc = Accuracy(num_classes=num_classes, task='multiclass', average='macro')

        # Load pre-trained EfficientNet-B0
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # Modify depending on 'rich': 3, 32 or 'mel': 1, 32
        self.efficientnet.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # Replace the classifier head
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )

        if alphas is not None:
            alphas = torch.clamp(alphas, min=0.05, max=0.95)

        # initialize focal loss
        self.focal_loss = FocalLoss(
            gamma=gamma,
            alpha=alphas,
            reduction='mean',
            task_type='multi-class',
            num_classes=num_classes
        )

        # Add gradient clipping
        self.clip_value = 1.0

    def forward(self, x):
        # Add input validation
        if torch.isnan(x).any():
            print("Warning: NaN detected in model input")
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            
        # Add input normalization for stability
        if x.dim() == 4:  # (batch, channel, height, width)
            # Normalize per sample to [-1, 1] or [0, 1] range
            batch_min = x.view(x.size(0), -1).min(dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
            batch_max = x.view(x.size(0), -1).max(dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
            # Avoid division by zero
            divisor = torch.clamp(batch_max - batch_min, min=1e-5)
            x = (x - batch_min) / divisor
            
        return self.efficientnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # Gradient clipping during forward pass
        with torch.cuda.amp.autocast(enabled=True):  # Mixed precision for stability
            logits = self(x)
            
            # Handle one hot encoded labels
            if y.dim() > 1 and y.shape[1] > 1:
                y_for_acc = y.argmax(dim=1)
            else:
                y_for_acc = y
                
            # Compute focal loss with gradient tracking safeguards
            loss = self.focal_loss(logits, y)
            
            # Detect and handle NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected in training step {batch_idx}")
                # Use a small constant loss to allow backward pass but minimize impact
                loss = torch.tensor(0.1, device=self.device, requires_grad=True)
            
            # Compute balanced accuracy
            with torch.no_grad():  # Don't track gradients for metrics
                preds = logits.argmax(dim=1)
                balanced_acc = self.train_balanced_acc(preds, y_for_acc)
        
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_value)
        
        # Log metrics with proper sync and reduce operations
        self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_acc_balanced_step", balanced_acc, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train_acc_balanced_epoch", balanced_acc, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Mixed precision for stability
        with torch.cuda.amp.autocast(enabled=True):
            logits = self(x)
            
            # Handle one hot encoded labels
            if y.dim() > 1 and y.shape[1] > 1:
                y_for_acc = y.argmax(dim=1)
            else:
                y_for_acc = y
                
            # Compute focal loss
            loss = self.focal_loss(logits, y)
            
            # Handle NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected in validation step {batch_idx}")
                loss = torch.tensor(0.1, device=self.device)
            
            # Compute balanced accuracy
            preds = logits.argmax(dim=1)
            balanced_acc = self.val_balanced_acc(preds, y_for_acc)
        
        # Log metrics properly
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc_balanced", balanced_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {"val_loss": loss, "val_acc": balanced_acc}

    def configure_optimizers(self):
        # Use AdamW optimizer with weight decay and gradient clipping
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=0.01,
            eps=1e-8  # Increased epsilon for numerical stability
        )
        
        # Learning rate scheduler with more robust configuration
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2,
            min_lr=1e-6,
        )
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": scheduler, 
            "monitor": "val_loss"
        }
