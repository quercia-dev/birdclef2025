from utility_data import *
from utility_plots import *

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torchmetrics.classification import Accuracy
import time


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


class MelCNN(pl.LightningModule):
    def __init__(self, num_classes: int, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.val_balanced_acc = Accuracy(num_classes=num_classes, task='multiclass', average='macro')
        self.train_balanced_acc = Accuracy(num_classes=num_classes, task='multiclass', average='macro')

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
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        probabilities = self(x)
        loss = F.cross_entropy(probabilities, y)
        balanced_acc = self.train_balanced_acc(probabilities.argmax(dim=1), y.argmax(dim=1))
        
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc_balanced", balanced_acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        probabilities = self(x)
        loss = F.cross_entropy(probabilities, y)
        balanced_acc = self.val_balanced_acc(probabilities.argmax(dim=1), y.argmax(dim=1))
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc_balanced", balanced_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.2)
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

    train_loader, val_loader = create_dataloaders(dataset)
    print("Constructed training data infrastructure")

    model = MelCNN(num_classes=len(dataset.classes))

    # tb_logger = TensorBoardLogger('model/tb_logs', name='melcnn')
    logger = CSVLogger("model/csv_logs", name="melcnn")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=-1,
        filename='{epoch}-{val_loss:.2f}',
        dirpath='./model/checkpoints',
        every_n_epochs=1
    )
    trainer = pl.Trainer(
        max_epochs=5, 
        callbacks=[checkpoint_callback], 
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