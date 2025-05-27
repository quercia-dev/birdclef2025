import torch
import torch.nn as nn 
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetAudio(pl.LightningModule):
    def __init__(self, num_classes: int):
        super().__init__()
        self.save_hyperparameters()
        self.val_acc = Accuracy(num_classes=num_classes, task='multiclass')
        self.train_acc = Accuracy(num_classes=num_classes, task='multiclass') 
        # Add balanced accuracy metrics
        self.val_balanced_acc = Accuracy(num_classes=num_classes, task='multiclass', average='macro')
        self.criterion = nn.CrossEntropyLoss()
        
        # Load pre-trained EfficientNet-B0
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        self.efficientnet.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # Replace the classifier head
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )


    def forward(self, x):
        return self.efficientnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = self.criterion(logits, y)
        acc = self.train_acc(logits.argmax(dim=1), y.argmax(dim=1))
                
        # Log metrics with proper sync and reduce operations
        self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_acc_step", acc, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train_acc_epoch", acc, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = self.criterion(logits, y)
        val_acc = self.val_acc(logits.argmax(dim=1), y.argmax(dim=1))
        val_balanced_acc = self.val_balanced_acc(logits.argmax(dim=1), y.argmax(dim=1))
        
        # Log metrics properly
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_balanced_acc", val_balanced_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {"val_loss": loss, "val_acc": val_acc, "val_balanced_acc": val_balanced_acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=1e-3, 
            weight_decay=0.01,
        )
        
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
