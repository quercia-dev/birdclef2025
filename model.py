# %%
from utility_data import *
from utility_plots import *

# %%
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import time

print("Imported libraries")

# %%
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
dataset = AudioDataset(
    datafolder="data",
    metadata_csv="train_proc.csv",
    audio_dir="train_proc",
    feature_mode=''
)

print(dataset[0][0].size())

print("Initialised Dataset")

# %% [markdown]
# ## Build a Model

# %%
class MelCNN(pl.LightningModule):
    def __init__(self, num_classes: int, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

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
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
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
    
print("Defined Model")

# %% [markdown]
# ## Training

# %%
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=7, persistent_workers=True)
val_loader = DataLoader(val_set, batch_size=32, num_workers=7, persistent_workers=True)
print("Constructed training data infrastructure")

# %%
model = MelCNN(num_classes=len(dataset.classes))

logger = TensorBoardLogger('tb_logs', name='melcnn')

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=3,
    filename='{epoch}-{val_loss:.2f}'
)
trainer = pl.Trainer(
    max_epochs=10, 
    callbacks=[checkpoint_callback], 
    logger=logger, 
    log_every_n_steps=10)

# %%
print("Beginning training")

start = time.time()
trainer.fit(model, train_loader, val_loader)
print(f"Execution time: {time.time() - start:.4f} seconds")

print("Finished training")


