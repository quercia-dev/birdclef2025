# %%
print("Importing")

from utility_data import *

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import time

print("Done importing")

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# %% [markdown]
# ## Prepare Data

# %%
# maybe go higher (start at 44.1 kHz, go higher 48-96 kHz if needed)
audio_params = {'sample_rate': 44000, 'n_fft': 1024, 'hop_length': 256, 'n_mfcc': 64, 'n_mels': 64, 'feature_size': 1024} # 316?

dataset = AudioDataset(
    datafolder="data",
    metadata_csv="train.csv",
    audio_dir="train_audio",
    extract_features=True,
    audio_params=audio_params
)

print("Initialised objects")

# %% [markdown]
# ## Build a Model

# %%
class MelCNN(pl.LightningModule):
    def __init__(self, num_classes: int, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [B, 3, 128, 316]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),            # [B, 32, 64, 158]
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B, 64, 64, 158]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),            # [B, 64, 32, 79]
            nn.Dropout(0.2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1), # [B, 64, 32, 79]
            nn.ReLU(),
        )

        flattened_size = 64 * 32 * 128

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

print("Defined model")

# %% [markdown]
# ## Training

# %%
print("Composing train data")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])

print("Loading Data and training model")

start = time.time()

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=7)
val_loader = DataLoader(val_set, batch_size=32, num_workers=7)

model = MelCNN(num_classes=len(dataset.classes))

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=3,
    filename='{epoch}-{val_loss:.2f}'
)
trainer = pl.Trainer(
    max_epochs=10, 
    callbacks=[checkpoint_callback],
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=2,
    precision='16-mixed'
)
trainer.fit(model, train_loader, val_loader)

print("Finished training")
print(f"Execution time: {time.time() - start:.4f} seconds")
