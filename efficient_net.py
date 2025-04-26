import torch
import torch.nn as nn 
import pytorch_lightning as pl
import torch.nn.functional as F 
from torchmetrics.classification import Accuracy
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
            nn.Linear(num_ftrs, num_classes),
            nn.Softmax(dim=1)
        )
        
        self.alphas = alphas
        self.gamma = gamma

    def forward(self, x):
        return self.efficientnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        probabilities = self(x)  # logits expected
        log_probs = F.log_softmax(probabilities, dim=-1)  # [batch_size, num_classes]
        weighted_log_probs = log_probs * self.alphas  # self.class_weights: [num_classes]
        loss = -(y * weighted_log_probs).sum(dim=-1).mean()

        balanced_acc = self.train_balanced_acc(probabilities.argmax(dim=1), y.argmax(dim=1))
        
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc_balanced", balanced_acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        probabilities = self(x)  # logits expected
        log_probs = F.log_softmax(probabilities, dim=-1)  # [batch_size, num_classes]
        weighted_log_probs = log_probs * self.alphas
        loss = -(y * weighted_log_probs).sum(dim=-1).mean()

        balanced_acc = self.val_balanced_acc(probabilities.argmax(dim=1), y.argmax(dim=1))
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc_balanced", balanced_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # Two-phase training strategy:
        # 1. Only train the classifier head for a few epochs
        # 2. Fine-tune the entire network

        # Initially freeze the backbone and only train the classifier
        # for param in self.efficientnet.features.parameters():
        #     param.requires_grad = False

        # Option 1: Use regular optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.gamma)
        
        # Option 2: Different learning rates for feature extractor and classifier
        # optimizer = torch.optim.AdamW([
        #     {'params': self.efficientnet.features.parameters(), 'lr': self.hparams.learning_rate * 0.1},
        #     {'params': self.efficientnet.classifier.parameters(), 'lr': self.hparams.learning_rate}
        # ], weight_decay=0.01)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
