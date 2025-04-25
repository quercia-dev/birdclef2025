from utility_data import *
from utility_plots import *
from focal_loss import FocalLoss

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torchmetrics.classification import Accuracy
import time
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def check_cuda():
    """
    Checks if CUDA (GPU support) is available and prints the device being used.
    """
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")


class EfficientNetAudio(pl.LightningModule):
    def __init__(self, num_classes: int, gamma: int, alpha: float, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.floss = FocalLoss(gamma=gamma, alpha=alpha, task_type='multi-label')
        
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

    def forward(self, x):
        return self.efficientnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        probabilities = self(x)
        loss = self.floss(probabilities, y)
        balanced_acc = self.train_balanced_acc(probabilities.argmax(dim=1), y.argmax(dim=1))
        
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc_balanced", balanced_acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        probabilities = self(x)
        loss = self.floss(probabilities, y)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)
        
        # Option 2: Different learning rates for feature extractor and classifier
        # optimizer = torch.optim.AdamW([
        #     {'params': self.efficientnet.features.parameters(), 'lr': self.hparams.learning_rate * 0.1},
        #     {'params': self.efficientnet.classifier.parameters(), 'lr': self.hparams.learning_rate}
        # ], weight_decay=0.01)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    

def create_dataloaders(dataset, batch_size=32, val_split=0.2, num_workers=7):
    """
    Splits the dataset into training and validation sets and returns their DataLoaders.

    Args:
        dataset (Dataset): The full dataset to split.
        batch_size (int): Number of samples per batch.
        val_split (float): Proportion of the dataset to use for validation.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size,
        num_workers=num_workers, persistent_workers=True
    )

    return train_loader, val_loader    

if __name__ == '__main__':
    check_cuda()
    
    dataset = AudioDataset(
        datafolder="data",
        metadata_csv="train_proc.csv",
        audio_dir="train_proc",
        feature_mode='mel'
    )
    
    # use 20% of the total dataset
    # dataset.data = dataset.data.sample(frac=0.2, random_state=42)

    train_loader, val_loader = create_dataloaders(dataset)
    print("Constructed training data infrastructure")

    model = EfficientNetAudio(num_classes=len(dataset.classes),
                              gamma=2,
                              alpha=0.25)

    # tb_logger = TensorBoardLogger('model/tb_logs', name='efficientnet_audio')
    logger = CSVLogger("model/csv_logs", name="efficientnet_audio")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=3,  # Save the top 3 models
        filename='{epoch}-{val_loss:.2f}',
        dirpath='./model/efficientnet_checkpoints',
        every_n_epochs=1
    )
    
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
    )
    
    trainer = pl.Trainer(
        max_epochs=20, 
        callbacks=[checkpoint_callback, early_stop_callback], 
        logger=logger, 
        log_every_n_steps=10)

    print("Beginning training")

    start = time.time()
    trainer.fit(model, train_loader, val_loader)

    elapsed_time = time.time() - start
    hours = elapsed_time // 3600
    minutes = (elapsed_time % 3600) // 60
    seconds = elapsed_time % 60
    print(f"Execution time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.4f} seconds")

    print("Finished training")