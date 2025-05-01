from utility_data import *

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter
import ast
from torch.utils.data import Dataset
from typing import Optional
import scipy.ndimage

def count_values_in_lists(df, column_name):
    all_values = []
    
    for item in df[column_name]:
        if isinstance(item, list):
            all_values.extend(item)
        else:
            try:
                parsed_list = ast.literal_eval(item)
                all_values.extend(parsed_list)
            except Exception:
                continue
    
    return pd.Series(Counter(all_values)).sort_values(ascending=False)


def sand_plot(data: pd.DataFrame, title:str="Sand graph", includeX: bool=False):
    _, ax = plt.subplots(figsize=(15, 5))
    data.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
    ax.set_yscale('log')

    ax.set_xlabel('primary_label')
    ax.set_ylabel('Values (log scale)')
    ax.set_title(title)
    ax.get_legend().set_visible(False)
    plt.xticks(fontsize=9)

    if not includeX:
        ax.set_xticklabels([])
    plt.show()


def plot_value_counts(data: pd.Series):
    values = data.value_counts()

    plt.figure(figsize=(10, 6))
    plt.scatter(values.index, values.values, color='blue', alpha=0.7)

    plt.title('Scatter Plot of Audio Duration Value Counts (Log Scale)')
    plt.xlabel('Audio Duration (seconds) [log scale]')
    plt.ylabel('Frequency [log scale]')

    plt.xscale('log')
    plt.yscale('log')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.show()

def format_duration(x, pos):
    minutes = int(x // 60)
    seconds = int(x % 60)
    return f'{minutes}:{seconds:02d}'

def plot_value_histograms(data: pd.Series, bins:int=200):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color='blue', alpha=0.7, log=True)

    plt.title('Histogram of Audio Duration (Log Scale)')
    plt.xlabel('Audio Duration (seconds)')
    plt.ylabel('Frequency (log scale)')

    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_duration))

    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.show()


def spectrogram(dataset: AudioDataset, index: int, transform=None, clusters: Optional[pd.Series] = None, cmap='tab10'):
    waveform, _ = dataset[index]
    
    if transform is not None:
        waveform = transform(waveform)
    
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    spec = pd.DataFrame(waveform.log2().detach().squeeze(0).numpy())
    
    _, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(spec, aspect='auto', origin='lower')
    plt.title(f"Spectrogram {dataset.classes[index]} (Index: {index})")
    plt.colorbar(im, ax=ax)

    # Optional: overlay color bands based on clustering
    if clusters is not None:
        if not isinstance(clusters, pd.Series):
            raise ValueError("clusters must be a pandas Series")
        
        x_len = spec.shape[1]
        cluster_labels = clusters.reindex(range(x_len), method='nearest')
        cluster_labels = cluster_labels.bfill().ffill().astype(int)

        unique_clusters = sorted(cluster_labels.unique())
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(unique_clusters)))
        color_map = dict(zip(unique_clusters, colors))
        
        for x in range(x_len):
            cluster_id = cluster_labels.iloc[x]
            ax.axvspan(x - 0.5, x + 0.5, ymin=0, ymax=0.1, color=color_map[cluster_id], linewidth=0)

    plt.tight_layout()
    plt.show()

def plot_training_log(csv_path):
    # Read CSV
    df = pd.read_csv(csv_path)

    # Plot
    _, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Accuracy
    axs[0].plot(df['step'], df['train_acc'], label='Train Accuracy', marker='o')
    axs[0].plot(df['step'], df['val_acc'], label='Val Accuracy', marker='x', linestyle='--')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('Training and Validation Accuracy')
    axs[0].legend()
    axs[0].grid(True)

    # Loss
    axs[1].plot(df['step'], df['train_loss'], label='Train Loss', marker='o')
    axs[1].plot(df['step'], df['val_loss'], label='Val Loss', marker='x', linestyle='--')
    axs[1].set_xlabel('Step')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Training and Validation Loss')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def summarize_secondary_labels(data: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes the counts of secondary labels grouped by each primary label.

    Parameters:
        data (pd.DataFrame): DataFrame containing at least 'primary_label' and 'secondary_labels' columns.

    Returns:
        pd.DataFrame: Summary DataFrame with secondary label counts per primary label.
    """
    primary_labels = data["primary_label"].unique()
    summary = pd.DataFrame(primary_labels, columns=["primary_label"])

    count_data = []
    for col in primary_labels:
        counts_dict = dict(count_values_in_lists(
            data[data["primary_label"] == col],
            "secondary_labels"
        ))
        count_data.append({"primary_label": col, **counts_dict})

    count_df = pd.DataFrame(count_data)
    summary = pd.merge(summary, count_df, on="primary_label", how="left", validate="one_to_one")

    summary = summary.fillna(0)
    summary = summary.rename(columns={'': 'empty'})

    summary['sum_values'] = summary.iloc[:, 1:].sum(axis=1)
    summary = summary.sort_values(by='sum_values', ascending=False)

    numeric_data = summary.iloc[:, 1:-1]
    numeric_data.index = summary["primary_label"]

    return summary, numeric_data
