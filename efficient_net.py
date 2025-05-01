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
        # Use pretrained weights for better transfer learning
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # Modify the first convolutional layer to accept 1-channel input (grayscale mel spectrograms)
        # instead of the default 3-channel RGB images
        self.efficientnet.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # Replace the classifier head
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )

        # initialize focal loss
        self.focal_loss = FocalLoss(
            gamma=gamma,
            alpha=alphas,
            reduction='mean',
            num_classes=num_classes
        )

    def forward(self, x):
        return self.efficientnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # handle one hot encoded labels
        if y.dim() > 1 and y.shape[1] > 1:
            y_for_acc = y.argmax(dim=1)
        else:
            y_for_acc = y

        # compute focal loss
        loss = self.focal_loss(logits, y)

        # compute balanced accuracy
        preds = logits.argmax(dim=1)
        balanced_acc = self.train_balanced_acc(preds, y_for_acc)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc_balanced", balanced_acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # handle one hot encoded labels
        if y.dim() > 1 and y.shape[1] > 1:
            y_for_acc = y.argmax(dim=1)
        else:
            y_for_acc = y

        # compute focal loss
        loss = self.focal_loss(logits, y)

        # compute balanced accuracy
        preds = logits.argmax(dim=1)
        balanced_acc = self.train_balanced_acc(preds, y_for_acc)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc_balanced", balanced_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # use AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)
        
        # learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
