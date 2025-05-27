from utility_data import *
from utility_plots import *
from efficient_net import EfficientNetAudio
from melcnn import MelCNN

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

import time
import argparse
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
    g = torch.Generator().manual_seed(25) # fixed seed for reproducibility
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=g)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size,
        num_workers=num_workers, persistent_workers=True
    )

    return train_loader, val_loader    


def train_model(results_folder:str, args, train_loader, val_loader, num_classes: int):
    model_descr = f'{args.model}'

    if args.model == 'efficient':
        model = EfficientNetAudio(num_classes=num_classes)
    elif args.model == 'melcnn':
        model = MelCNN(num_classes=num_classes)

    logsdir = os.path.join(results_folder, f'{args.log}_logs')
    # Select logger
    if args.log == 'tensor':
        logger = TensorBoardLogger(logsdir, name=model_descr)
    elif args.log == 'csv':
        logger = CSVLogger(logsdir)
    else:
        raise ValueError(f'Unrecognized logger option: {args.log}')
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        filename='{epoch}-{val_loss:.2f}',
        dirpath=os.path.join(results_folder, 'checkpoints'),
        every_n_epochs=1
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs, 
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
    best_val_balanced_acc = float(trainer.callback_metrics.get("val_balanced_acc", 0.0))

    return best_val_loss, best_val_acc, best_val_balanced_acc

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

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train the model for: 10 is default')

    parser.add_argument('--dataset', choices=['labelled', 'yamnet', 'yamnet_light'], default='labelled',
                        help='Choose the data to train the model on: labelled or yamnet')

    parser.add_argument('--mass', type=float, default=1.0,
                        help='Set importance m ∈ [0,1] for primary labels; 1.0 means one-hot encoding')

    args = parser.parse_args()
    
    print('Running arguments: ', args)

    device = check_cuda()
    
    dataset = AudioDataset(
        datafolder="data",
        metadata_csv="train_proc.csv",
        audio_dir="train_proc",
        feature_mode='mel',
        m=args.mass
    )
    if args.dataset == 'yamnet':
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
        dataset.filter_by_values('yamnet', selected_labels)

    elif args.dataset == 'yamnet_light':
        selected_labels = ['Silence', 'Speech', 'Vehicle']
        dataset.filter_by_values('yamnet', selected_labels, include=False)
    
    train_loader, val_loader = create_dataloaders(dataset)
    print("Constructed training data infrastructure")

    results_folder = f'./model/{time.strftime("%Y%m%d_%H%M%S")}_{args.model}_{args.dataset}_{args.mass}'
    
    if os.path.exists(results_folder):
        shutil.rmtree(results_folder)
    os.makedirs(results_folder)
    
    best_val_loss, best_val_acc, best_val_balanced_acc = train_model(results_folder, 
                                              args, 
                                              train_loader, 
                                              val_loader, 
                                              num_classes=len(dataset.classes))
    
    print(f'Final Results:')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    print(f'Best validation balanced accuracy: {best_val_balanced_acc:.4f}')