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
from torch.utils.data import DataLoader
from torchvision import transforms

from model.RTFNet import RTFNet
from util.RGBTDataset import RGBThermalDataset


import os

# Set the environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
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


def plot_loss_curves(train_losses, val_losses, pid):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f"P{pid}: Training and Validation Loss Curves".format(pid=pid))
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(labels, preds, class_names, pid):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'P{pid} Confusion Matrix'.format(pid=pid))
    plt.show()


def train(model, train_loader, criterion, optimizer, device):
    model.train()  # Set the model to training mode
    total_loss = 0.0

    for batch_idx, (rgb_images, thermal_images, labels, lengths) in enumerate(train_loader):

        if rgb_images is not None and thermal_images is not None and labels is not None and lengths is not None:
            optimizer.zero_grad()
            rgb_images = rgb_images.to(device)
            thermal_images = thermal_images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(rgb_images, thermal_images, lengths)

            # Compute loss
            loss = criterion(outputs, labels)
            x = loss.item()
            total_loss += x

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            print (f"{batch_idx}/{len(train_loader)}: {x}")

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation
        for batch_idx, (rgb_images, thermal_images, labels, lengths) in enumerate(val_loader):
            rgb_images = rgb_images.to(device)
            thermal_images = thermal_images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(rgb_images, thermal_images, lengths)

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


def test(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():  # Disable gradient calculation
        for batch_idx, (rgb_images, thermal_images, labels, lengths) in enumerate(test_loader):
            # Move data to the appropriate device
            rgb_images = rgb_images.to(device)
            thermal_images = thermal_images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(rgb_images, thermal_images, lengths)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

    return all_labels, all_preds, accuracy, precision, recall, f1


def main(params, pid, mode='single_inference'):
    # For reproducibility
    set_random_seed(42)
    g = torch.Generator()
    g.manual_seed(0)

    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    num_classes = 3
    num_workers = params['num_workers']
    device_ids = [1, 0, 2, 3]  # List of GPUs to use
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Initialize the model
    model = RTFNet(n_class=num_classes,
                   num_resnet_layers=params['num_resnet_layers'],
                   num_lstm_layers=params['num_lstm_layers'],
                   lstm_hidden_size=params['lstm_hidden_size'],
                   device=device)
    # Wrap the model with DataParallel
    model = nn.DataParallel(model, device_ids=device_ids)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ])
    # Create datasets
    data_dir = params['data_dir']
    train_dataset = RGBThermalDataset(data_dir=data_dir, pid=pid, split='train', transform=transform)
    val_dataset = RGBThermalDataset(data_dir=data_dir, pid=pid, split='val', transform=transform)
    test_dataset = RGBThermalDataset(data_dir=data_dir, pid=pid, split='test', transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             collate_fn=collate_fn, pin_memory=True)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = params['early_stop_patience']
    early_stop_count = 0

    interval = max(params['num_epochs'] // 5, 1)  # Ensure at least 1

    for epoch in range(num_epochs):
        # print(f'Epoch {epoch + 1}/{num_epochs}')

        # Train the model
        train_loss = train(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validate the model
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(f"P{pid}, Epoch {epoch + 1}/{params['num_epochs']}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
        # if mode != 'grid_search' and (epoch + 1) % interval == 0:
        #     print(f"P{pid}, Epoch {epoch + 1}/{params['num_epochs']}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
            # Save model if needed
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print(f"P{pid}, Early stopping triggered. Stopped at epoch {epoch + 1}/{params['num_epochs']}, Train Loss: {train_loss}, Validation Loss: {val_loss}.")
                break

    # Plot loss curve
    if mode != 'grid_search':
        plot_loss_curves(train_losses, val_losses, pid)

    # Testing
    all_labels, all_preds, accuracy, precision, recall, f1 = test(model, test_loader, device)

    # Plot confusion matrix
    if mode != 'grid_search':
        class_names = ['Negative', 'Smoking', 'Eating']
        plot_confusion_matrix(all_labels, all_preds, class_names, pid)
    return precision, recall, accuracy, f1


def lopo_train(params):
    participant_pids = [6, 7, 13, 14, 15, 16, 18]
    metrics_dict = {
        'participant_id': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'accuracy': []
    }
    for pid in participant_pids:
        precision, recall, accuracy, f1 = main(params, pid)
        metrics_dict['participant_id'].append(pid)
        metrics_dict['precision'].append(round(precision, 4))
        metrics_dict['recall'].append(round(recall, 4))
        metrics_dict['f1_score'].append(round(f1, 4))
        metrics_dict['accuracy'].append(round(accuracy, 4))
        print(f"Precision: {round(precision, 4)}, Recall: {round(recall, 4)}, F1 score: {round(f1, 4)}, Accuracy: {round(accuracy, 4)}")
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.set_index('participant_id', inplace=True)
    avg_metrics = metrics_df.mean().round(4).to_frame().T
    avg_metrics.index = ['Mean']
    metrics_df = pd.concat([metrics_df, avg_metrics])
    print(metrics_df)


if __name__ == '__main__':
    params = {
        'num_workers': 4,  # For data loading
        'num_resnet_layers': 18,
        'num_lstm_layers': 1,
        'lstm_hidden_size': 128,
        'num_epochs': 10,
        'batch_size': 1,
        'learning_rate': 0.00001,
        'data_dir': "/ssd1/meixi/data",
        'early_stop_patience': 5,
    }
    lopo_train(params)
