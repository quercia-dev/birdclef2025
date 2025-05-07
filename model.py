from utility_data import *
from utility_plots import *
from efficient_net import EfficientNetAudio

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torchmetrics.classification import Accuracy

import time
import argparse
import csv
import shutil

def check_cuda():
    """
    Checks if CUDA (GPU support) is available and prints the device being used.
    Returns:
        torch.device: The device to use (GPU or CPU)
    """
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


class MelCNN(pl.LightningModule):
    def __init__(self, num_classes: int):
        super().__init__()
        self.save_hyperparameters()
        
        self.val_acc = Accuracy(num_classes=num_classes, task='multiclass')
        self.train_acc = Accuracy(num_classes=num_classes, task='multiclass')

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
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x) # [batch_size, num_classes]
        
        # Convert one-hot encoded targets to class indices if necessary
        if y.dim() > 1 and y.size(1) > 1:  # One-hot encoded
            y_idx = y.argmax(dim=1)
        else:
            y_idx = y
        
        loss = self.criterion(logits, y_idx)
        acc = self.train_acc(logits.argmax(dim=1), y_idx)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)  # [batch_size, num_classes]
        
        # Convert one-hot encoded targets to class indices if necessary
        if y.dim() > 1 and y.size(1) > 1:  # One-hot encoded
            y_idx = y.argmax(dim=1)
        else:
            y_idx = y
        
        loss = self.criterion(logits, y_idx)
        acc = self.val_acc(logits.argmax(dim=1), y_idx)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    

def create_dataloaders(dataset, batch_size=32, val_split=0.2, num_workers=2):
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


def train_model(results_folder:str, args, train_loader, val_loader):
    model_descr = f'{args.model}'

    if args.model == 'efficient':
        model = EfficientNetAudio(num_classes=len(dataset.classes))
    elif args.model == 'melcnn':
        model = MelCNN(num_classes=len(dataset.classes))

    # Select logger
    if args.log == 'tensor':
        logger = TensorBoardLogger('model/tb_logs', name=model_descr)
    elif args.log == 'csv':
        logger = CSVLogger("model/csv_logs", name=model_descr)
    else:
        logger = True

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        filename='{epoch}-{val_loss:.2f}',
        dirpath=os.path.join(results_folder, f'checkpoints/{model_descr}'),
        every_n_epochs=1
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
    )
    trainer = pl.Trainer(
        max_epochs=15, 
        callbacks=[checkpoint_callback, early_stop_callback], 
        logger=logger, 
        gradient_clip_val=1.0,  # Add gradient clipping
        gradient_clip_algorithm="norm",
        log_every_n_steps=10,
        enable_checkpointing=True,
        enable_progress_bar=True
        )

    print("Begin training", model_descr)

    start = time.time()
    trainer.fit(model, train_loader, val_loader)
    print_elapsed(time.time()-start, model_descr)
    
    best_val_loss = checkpoint_callback.best_model_score.item()
    best_val_acc = float(trainer.callback_metrics.get("val_acc", 0.0))

    return best_val_loss, best_val_acc

def print_elapsed(elapsed_time, descr:str = 'Model'):    
    hours = elapsed_time // 3600
    minutes = (elapsed_time % 3600) // 60
    seconds = elapsed_time % 60
    print(f"{descr}: {int(hours)} hours, {int(minutes)} minutes, {seconds:.4f} seconds")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Audio classification training script")
    parser.add_argument('--log', choices=['tensor', 'csv'], default='csv',
                        help='Choose the logger: "tensor" for TensorBoard, "csv" for CSV logger (default: csv)')
    parser.add_argument('--model', choices=['melcnn', 'efficient'], default='efficient',
                        help='Choose the model architecture: "melcnn" (default) or "efficient"')

    args = parser.parse_args()

    device = check_cuda()
    
    dataset = AudioDataset(
        datafolder="data",
        metadata_csv="train_proc.csv",
        audio_dir="train_proc",
        feature_mode='mel',
        m=1
    )
    # limits the dataset to relevant labels
    selected_labels = ['Animal', 
                        'Wild animals', 
                        'Insect', 
                        'Cricket',
                        'Bird', 
                        'Bird vocalization, bird call, bird song', 
                        'Frog', 
                        'Snake', 
                        'Whistling', 
                        'Owl', 
                        'Crow', 
                        'Rodents, rats, mice', 
                        'Livestock, farm animals, working animals', 
                        'Pig', 
                        'Squeak', 
                        'Domestic animals, pets', 
                        'Dog', 
                        'Turkey', 
                        'Bee, wasp, etc.', 
                        'Duck', 
                        'Chicken, rooster', 
                        'Horse', 
                        'Goose', 
                        'Squawk', 
                        'Chirp tone', 
                        'Sheep', 
                        'Pigeon, dove']

    dataset.data = dataset.data[dataset.data['yamnet'].isin(selected_labels)]
    
    train_loader, val_loader = create_dataloaders(dataset)
    print("Constructed training data infrastructure")

    results_folder = './model_filtered'
    
    if os.path.exists(results_folder):
        shutil.rmtree(results_folder)
    os.makedirs(results_folder)
    
    best_val_loss, best_val_acc = train_model(results_folder, args, train_loader, val_loader)
    
    print(f'best_val_loss {best_val_loss}, best_val_acc {best_val_acc}')