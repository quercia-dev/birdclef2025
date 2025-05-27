import torch
import torch.nn as nn 
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

class MelCNN(pl.LightningModule):
    def __init__(self, num_classes: int):
        super().__init__()
        self.save_hyperparameters()
        
        self.val_acc = Accuracy(num_classes=num_classes, task='multiclass')
        self.train_acc = Accuracy(num_classes=num_classes, task='multiclass')
        self.criterion = nn.CrossEntropyLoss()

        # input size: [B, 1, 128, 320]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B, 32, 128, 320]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),            # [B, 32, 64, 160]
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B, 64, 64, 160]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),            # [B, 64, 32, 80]
            nn.Dropout(0.2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1), # [B, 64, 32, 80]
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # global avg pool to avoid large FC layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),           # [B, 64]
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x) # [batch_size, num_classes]
                
        loss = self.criterion(logits, y)
        acc = self.train_acc(logits.argmax(dim=1), y.argmax(dim=1))
        
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch
            logits = self(x)  # [batch_size, num_classes]
            
            loss = self.criterion(logits, y)
            acc = self.train_acc(logits.argmax(dim=1), y.argmax(dim=1))
            
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}