import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter
import ast
from torch.utils.data import Dataset
from typing import Optional


def count_values_in_lists(df, column_name):
    all_values = []
    
    for item in df[column_name]:
        if isinstance(item, list):
            all_values.extend(item)
        else:
            try:
                parsed_list = ast.literal_eval(item)
                all_values.extend(parsed_list)
            except:
                continue
    
    return pd.Series(Counter(all_values)).sort_values(ascending=False)


def sand_plot(data: pd.DataFrame, title:str="Sand graph", includeX: bool=False):
    ig, ax = plt.subplots(figsize=(15, 5))
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


def spectrogram(dataset: Dataset, index: int, clusters: Optional[pd.Series] = None, cmap='tab10'):
    waveform, label = dataset[index]
    spec = waveform.squeeze(0).log2().detach().numpy()  # Shape: (freq_bins, time_frames)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(spec, aspect='auto', origin='lower')
    plt.title(f"Spectrogram {dataset.classes[label]} (Index: {index})")
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