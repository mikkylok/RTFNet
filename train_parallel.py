import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from model.RTFNet import RTFNet
from util.RGBTDataset import RGBThermalDataset
import os

# Set the environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


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
    rgb_images, thermal_images, labels, timestamps = zip(*batch)

    # Find sequence lengths
    lengths = [len(seq) for seq in rgb_images]

    # Pad sequences
    rgb_images = rnn_utils.pad_sequence(rgb_images, batch_first=True)
    thermal_images = rnn_utils.pad_sequence(thermal_images, batch_first=True)

    # Stack labels
    labels = torch.stack(labels)

    return rgb_images, thermal_images, labels, lengths


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


def train(rank, world_size, params, pid, output_dir):
    setup(rank, world_size)

    # For reproducibility
    set_random_seed(42)

    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    num_classes = 3
    num_workers = params['num_workers']

    # Map the rank to the correct GPU (2 or 3)
    device = torch.device(f'cuda:{rank + 1}')

    # Initialize the model and move it to the current device
    model = RTFNet(n_class=num_classes,
                   num_resnet_layers=params['num_resnet_layers'],
                   num_lstm_layers=params['num_lstm_layers'],
                   lstm_hidden_size=params['lstm_hidden_size'],
                   device=device).to(device)

    # Wrap the model with DDP
    model = DDP(model, device_ids=[rank + 1])

    # Criterion and Optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create datasets and Distributed Sampler
    data_dir = params['data_dir']
    train_dataset = RGBThermalDataset(data_dir=data_dir, pid=pid, split='train', transform=transform)
    val_dataset = RGBThermalDataset(data_dir=data_dir, pid=pid, split='val', transform=transform)
    test_dataset = RGBThermalDataset(data_dir=data_dir, pid=pid, split='test', transform=transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=True, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             collate_fn=collate_fn, pin_memory=True, sampler=test_sampler)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = params['early_stop_patience']
    early_stop_count = 0

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # Ensure all samples are used equally across all epochs
        model.train()

        # Training loop
        train_loss = 0.0
        num_batches = len(train_loader)
        for batch_idx, (rgb_images, thermal_images, labels, lengths) in enumerate(train_loader):
            optimizer.zero_grad()
            rgb_images = rgb_images.to(device)
            thermal_images = thermal_images.to(device)
            labels = labels.to(device)
            outputs = model(rgb_images, thermal_images, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            x = loss.item()
            train_loss += x
            # if num_batches > 100 and (batch_idx + 1) % 100 == 0:
                # print(f'Batch {batch_idx + 1}/{num_batches}, Loss: {x}')

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rgb_images, thermal_images, labels, lengths in val_loader:
                rgb_images = rgb_images.to(device)
                thermal_images = thermal_images.to(device)
                labels = labels.to(device)
                outputs = model(rgb_images, thermal_images, lengths)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Rank {rank}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print(f"Rank {rank}, Early stopping triggered at epoch {epoch + 1}.")
                break

    # Testing loop
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for rgb_images, thermal_images, labels, lengths in test_loader:
            rgb_images = rgb_images.to(device)
            thermal_images = thermal_images.to(device)
            labels = labels.to(device)
            outputs = model(rgb_images, thermal_images, lengths)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Only rank 0 saves the plots and prints the test results
    if rank == 0:
        plot_loss_curves(train_losses, val_losses, pid, output_dir)
        class_names = ['Negative', 'Smoking', 'Eating']
        plot_confusion_matrix(all_labels, all_preds, class_names, pid, output_dir)

        # Print test results
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        print(f"Test Results for P{pid}:, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Cleanup
    dist.destroy_process_group()


def lopo_train(params, world_size):
    participant_pids = [6, 7, 13, 14, 15, 16, 18]
    output_dir = "."  # Set your desired output directory here
    os.makedirs(output_dir, exist_ok=True)
    for pid in participant_pids:
        mp.spawn(train, args=(world_size, params, pid, output_dir), nprocs=world_size, join=True)


if __name__ == '__main__':
    params = {
        'num_workers': 16,
        'num_resnet_layers': 18,
        'num_lstm_layers': 1,
        'lstm_hidden_size': 128,
        'num_epochs': 6,
        'batch_size': 1,
        'learning_rate': 0.00001,
        'data_dir': "/ssd1/meixi/data",
        'early_stop_patience': 5,
    }
    world_size = 3  # Only use GPUs 2 and 3
    lopo_train(params, world_size)
