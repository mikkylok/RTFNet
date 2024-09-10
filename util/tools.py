import os
import glob
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.utils.rnn as rnn_utils
import torch.distributed as dist


# Set the environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
BASE_LR = 0.005
STEPS = [0, 11, 14]
LRS = [1, 0.1, 0.01]


def get_lr_at_epoch(cur_epoch, max_epoch):
    """
    Retrieves the lr step index for the given epoch.
    Args:
        cur_epoch (float): the number of epoch of the current training stage.
    """
    lr_steps = STEPS + [max_epoch]
    for ind, step in enumerate(lr_steps):
        if cur_epoch < step:
            break
    return LRS[ind - 1] * BASE_LR


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_fn(batch):
    rgb_images, thermal_images, labels, timestamps, rgb_dir, thermal_dir = zip(*batch)

    # Find sequence lengths
    lengths = [len(seq) for seq in rgb_images]

    # Pad sequences
    rgb_images = rnn_utils.pad_sequence(rgb_images, batch_first=True)
    thermal_images = rnn_utils.pad_sequence(thermal_images, batch_first=True)

    # Stack labels
    labels = torch.stack(labels)

    return rgb_images, thermal_images, labels, lengths, rgb_dir, thermal_dir


def plot_loss_curves(train_losses, val_losses, pid, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f"P{pid}: Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"P{pid}_loss_plot.png"))
    plt.close()


def plot_confusion_matrix(labels, preds, class_names, pid, output_dir):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'P{pid} Confusion Matrix')
    plt.savefig(os.path.join(output_dir, f"P{pid}_confusion_matrix.png"))
    plt.close()


def find_best_checkpoint(output_dir, pid):
    # Use glob to search for files matching the pattern "P<pid>_best_checkpoint_epoch_*.pth.tar"
    checkpoint_pattern = os.path.join(output_dir, f"P{pid}_best_checkpoint_epoch_*.pth.tar")
    checkpoint_files = glob.glob(checkpoint_pattern)

    if not checkpoint_files:
        return None  # No checkpoint found

    # Return the first matching checkpoint file
    return checkpoint_files[0]
