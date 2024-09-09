import time
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
from fvcore.nn import FlopCountAnalysis

from model.RTFNet import RTFNet
from util.RGBTDataset import RGBThermalDataset
import os

# Set the environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
BASE_LR = 0.005
STEPS = [0, 11, 14]
LRS = [1, 0.1, 0.01]


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


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


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
    # optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    optimizer = optim.SGD(
        model.parameters(),
        lr=params['learning_rate'],
        momentum=params['momentum'],
        weight_decay=params['weight_decay'],
        dampening=params['dampening'],
        nesterov=params['nesterov'],
    )

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
    best_checkpoint_path = None
    final_checkpoint_path = None
    # patience = params['early_stop_patience']
    # early_stop_count = 0

    for epoch in range(num_epochs):
        new_lr = get_lr_at_epoch(epoch, num_epochs)
        set_lr(optimizer, new_lr)
        start_time = time.time()
        train_sampler.set_epoch(epoch)  # Ensure all samples are used equally across all epochs
        model.train()

        # Training loop
        train_loss = 0.0
        num_batches = len(train_loader)
        for batch_idx, (rgb_images, thermal_images, labels, lengths, rgb_dirs, thermal_dirs) in enumerate(train_loader):
            optimizer.zero_grad()
            rgb_images = rgb_images.to(device)
            thermal_images = thermal_images.to(device)
            labels = labels.to(device)
            outputs, _, _ = model(rgb_images, thermal_images, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            x = loss.item()
            train_loss += x
            # if num_batches > 100 and (batch_idx + 1) % 100 == 0:
            #     print(f'Pid {pid}, Rank {rank}, Batch {batch_idx + 1}/{num_batches}, Train Loss: {x}')

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rgb_images, thermal_images, labels, lengths, rgb_dirs, thermal_dirs in val_loader:
                rgb_images = rgb_images.to(device)
                thermal_images = thermal_images.to(device)
                labels = labels.to(device)
                outputs, _, _ = model(rgb_images, thermal_images, lengths)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        epoch_time = (time.time() - start_time) / 60
        print(f"Pid {pid}, Rank {rank + 1}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Epoch Time: {epoch_time:.2f} minutes, Learning rate: {optimizer.param_groups[0]['lr']}")

        # Save the best model checkpoint after each epoch
        if rank == 0:  # Save only from rank 0 to avoid multiple saves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Delete the previous best checkpoint if it exists
                if best_checkpoint_path is not None and os.path.exists(best_checkpoint_path):
                    os.remove(best_checkpoint_path)
                # Save the new best model
                best_checkpoint_path = os.path.join(output_dir, f"P{pid}_best_checkpoint_epoch_{epoch + 1}.pth.tar")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, filename=best_checkpoint_path)

        # Early stopping
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     early_stop_count = 0
        # else:
        #     early_stop_count += 1
        #     if early_stop_count >= patience:
        #         print(f"Pid {pid}, Rank {rank + 1}, Early stopping triggered at epoch {epoch + 1}.")
        #         break

    # Save final checkpoint after training finishes
    if rank == 0:
        final_checkpoint_path = os.path.join(output_dir, f"P{pid}_final_checkpoint.pth.tar")
        final_checkpoint = {
            'epoch': num_epochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }
        save_checkpoint(final_checkpoint, filename=final_checkpoint_path)

    # Testing loop
    if rank == 0:
        # Load the best validation checkpoint
        if best_checkpoint_path is not None and os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} with validation loss: {checkpoint['best_val_loss']}")
        else:
            print("Best checkpoint not found, using the final model.")

        model.eval()
        all_labels = []
        all_preds = []
        results = []  # To store all results for saving in CSV
        with torch.no_grad():
            for batch_idx, (rgb_images, thermal_images, labels, lengths, rgb_dirs, thermal_dirs) in enumerate(test_loader):
                rgb_images = rgb_images.to(device)
                thermal_images = thermal_images.to(device)
                labels = labels.to(device)

                # Calculate FLOPs for the first batch
                if batch_idx == 0:
                    flops = FlopCountAnalysis(model, (rgb_images, thermal_images, lengths))
                    tflops = flops.total() / 1e12  # Convert FLOPs to TFLOPs
                    print(f"Estimated TFLOPs for a single forward pass: {tflops:.6f} TFLOPs")

                outputs, rgb_weights, thermal_weights = model(rgb_images, thermal_images, lengths)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                # Append to the results for each video clip
                for i in range(len(labels)):
                    result = {
                        'rgb_video_path': rgb_dirs[i],
                        'thermal_video_path': thermal_dirs[i],
                        'true_label': labels[i].cpu().item(),
                        'prediction': preds[i].cpu().item(),
                        'prob_max': probs[i].max().cpu().item(),
                        'prob_0': probs[i][0].cpu().item(),
                        'prob_1': probs[i][1].cpu().item(),
                        'prob_2': probs[i][2].cpu().item(),
                        'rgb_weight': rgb_weights[i].mean().cpu().item(),  # Assuming you want the mean weight
                        'thermal_weight': thermal_weights[i].mean().cpu().item()  # Assuming you want the mean weight
                    }
                    results.append(result)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # Only rank 0 saves the plots and prints the test results
        plot_loss_curves(train_losses, val_losses, pid, output_dir)
        class_names = ['Negative', 'Smoking', 'Eating']
        plot_confusion_matrix(all_labels, all_preds, class_names, pid, output_dir)

        # Print test results
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        print(f"Test Results for P{pid}:, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        # Convert results to a DataFrame
        df = pd.DataFrame(results)
        csv_file_path = os.path.join(output_dir, f"P{pid}_label_pred_prob.csv")
        df.to_csv(csv_file_path, index=False)

    # Cleanup
    dist.destroy_process_group()


def lopo_train(params, world_size, participant_pids, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for pid in participant_pids:
        mp.spawn(train, args=(world_size, params, pid, output_dir), nprocs=world_size, join=True)


if __name__ == '__main__':
    params = {
        'num_workers': 16,
        'num_resnet_layers': 50,
        'num_lstm_layers': 1,  # can be grid searched [1,2] trying
        'lstm_hidden_size': 1024,  # can be grid searched [256, 512, 768, 1024] 1024 can fit with batch_size=5
        'num_epochs': 15,
        'batch_size': 5,   # when batch_size=3, resize can not be removed  # batch_size=5 when there is resize
        'learning_rate': 0.005,  # can be grid searched [0.00001, 0.000001]
        'data_dir': "/home/meixi/data",
        'early_stop_patience': 5,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'dampening': 0,
        'nesterov': True,
    }
    world_size = 3  # Only use GPUs 1, 2 and 3
    output_dir = "/home/meixi/mid_fusion/rtfnet/output/no_attention_late_fusion_no_skip_connection"
    participant_pids = [6, 7, 13, 14, 15, 16, 18]
    lopo_train(params, world_size, participant_pids, output_dir)