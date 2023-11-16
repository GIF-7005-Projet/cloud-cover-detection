import torch
import numpy as np
import matplotlib.pyplot as plt


def show_rgb_infrared_ground_truth(X: torch.Tensor, y: torch.Tensor, max_rows: int = 10):
    """
    Inputs:
        X: shape (batch, channels, height, width) expected to be in the range [0, 1]
        y: shape (batch, height, width)
    """
    nb_rows = min(max_rows, X.shape[0])
    fig, axes = plt.subplots(nrows=nb_rows, ncols=3, figsize=(10, nb_rows*4))
    for i in range(nb_rows):
        # Because X channels are Blue, Green, Red, Near infrared light by default
        X_rgb = torch.cat((X[i, 2, :, :].unsqueeze(0), X[i, 1, :, :].unsqueeze(0), X[i, 0, :, :].unsqueeze(0)),axis=0)
        axes[i, 0].imshow(X_rgb.cpu().numpy().transpose(1, 2, 0))
        axes[i, 0].set_title(f"RGB {i}")
        axes[i, 0].axis('off')
        axes[i, 1].imshow(X[i, 3, :, :].cpu().numpy())
        axes[i, 1].set_title(f"Near infrared light {i}")
        axes[i, 1].axis('off')
        axes[i, 2].imshow(y[i].cpu().numpy())
        axes[i, 2].set_title(f"Ground truth {i}")
        axes[i, 2].axis('off')
    plt.tight_layout()
    plt.show()


def show_channels(X: torch.Tensor, max_rows: int = 10):
    """
    Inputs:
        X: shape (batch, channels, height, width) expected to be in the range [0, 1]
    """
    nb_rows = min(max_rows, X.shape[0])
    _, axes = plt.subplots(nrows=nb_rows, ncols=X.shape[1], figsize=(10, nb_rows*2.5))
    for i in range(nb_rows):
        for channel in range(X.shape[1]):
            axes[i, channel].imshow(X[i, channel, :, :].cpu().numpy())
            axes[i, channel].axis('off')
    plt.tight_layout()
    plt.show()


def show_stats(X: torch.Tensor):
    """
    Inputs: shape (batch, channels, height, width)
    """
    X_float = X.float()
    channels_min = X.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]# shape (batch, channels)
    channels_max = X.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]# shape (batch, channels)
    channels_mean = X_float.mean(dim=(2, 3))# shape (batch, channels)
    channels_std = X_float.std(dim=(2, 3))# shape (batch, channels)

    channels = X.shape[1]

    _, axes = plt.subplots(nrows=channels, ncols=1, figsize=(10, channels * 4))

    if channels == 1:
        axes = [axes]

    for i in range(channels):
        ax = axes[i]

        channel_data = X[:, i, :, :].flatten().cpu().numpy()
        
        ax.hist(channel_data, bins=100, density=True, color=f'C{i}', alpha=0.75)

        ax.set_title(f'Channel {i} distribution')
        ax.set_xlabel('Pixel value')
        ax.set_ylabel('Frequency')

        stats_str = (f'Min: {channels_min[:, i].min():.2f}\n'
                     f'Max: {channels_max[:, i].max():.2f}\n'
                     f'Mean: {channels_mean[:, i].mean():.2f}\n'
                     f'Std: {channels_std[:, i].mean():.2f}')
        ax.text(0.95, 0.95, stats_str, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right')

    plt.tight_layout()
    plt.show()


def show_class_distribution(y: torch.Tensor):
    """
    Inputs: shape (batch, height, width)
    """
    class_counts = np.bincount(y.flatten().cpu().numpy())
    class_counts = class_counts / class_counts.sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(class_counts)), class_counts)
    ax.set_xticks(np.arange(len(class_counts)))
    ax.set_xticklabels(np.arange(len(class_counts)))
    ax.set_xlabel('Class')
    ax.set_ylabel('Frequency')
    ax.set_title('Class distribution')
    plt.show()
