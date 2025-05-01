from utility_data import *
from utility_plots import *
from focal_loss import FocalLoss
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
    def __init__(self, num_classes: int, gamma:int, alphas:torch.Tensor, learning_rate: float = 1e-3):
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
        
        self.alphas = alphas
        self.gamma = gamma

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        return self.classifier(x)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.gamma)
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


def train_model(results_folder:str, args, gamma:float, alphas:list, train_loader, val_loader):
    model_descr = f'{args.model}_g{gamma}'

    if args.model == 'melcnn':
        model = MelCNN(num_classes=len(dataset.classes), gamma=gamma, alphas=alphas)
    elif args.model == 'efficient':
        model = EfficientNetAudio(num_classes=len(dataset.classes), gamma=gamma, alphas=alphas, learning_rate=1e-3)

    # Select logger
    if args.log == 'tensor':
        logger = TensorBoardLogger('model/tb_logs', name=model_descr)
    elif args.log == 'csv':
        logger = CSVLogger("model/csv_logs", name=model_descr)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
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
        max_epochs=20, 
        callbacks=[checkpoint_callback, early_stop_callback], 
        logger=logger, 
        log_every_n_steps=10)

    print("Begin training", model_descr)

    start = time.time()
    trainer.fit(model, train_loader, val_loader)
    print_elapsed(time.time()-start, model_descr)
    
    best_val_loss = checkpoint_callback.best_model_score.item()
    best_val_acc = float(trainer.callback_metrics.get("val_acc_balanced", 0.0))

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
    parser.add_argument('--model', choices=['melcnn', 'efficient'], default='melcnn',
                        help='Choose the model architecture: "melcnn" (default) or "efficient"')

    args = parser.parse_args()

    device = check_cuda()
    
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

    results_folder = './model'
    
    if os.path.exists(results_folder):
        shutil.rmtree(results_folder)
    os.makedirs(results_folder)
    
    results_file = os.path.join(results_folder, 'grid_search.csv')
    with open(results_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['gamma', 'null_alpha', 'best_val_loss', 'best_val_acc'])  # header

    decay = [0, 0.1, 1, 3]
    null_alpha = [0] # importance of the null vector in CE loss
    for alpha in null_alpha:
        for dec in decay:
            alpha_t = torch.tensor([alpha], dtype=torch.float32)
            alphas_augmented = torch.cat([alpha_t, dataset.alphas], dim=0).to(device)
            best_val_loss, best_val_acc = train_model(results_folder, args, dec, alphas_augmented, train_loader, val_loader)

            with open(results_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([dec, alpha, best_val_loss, best_val_acc])
                
            print(f"Logged results for gamma={dec}, null_alpha={alpha}, val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f}")