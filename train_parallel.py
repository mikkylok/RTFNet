import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
import torch.distributed as dist


from model.RTFNet import RTFNet
from util.RGBTDataset import RGBThermalDataset
from util.tools import setup, set_random_seed, collate_fn, get_lr_at_epoch, set_lr
from test_parallel import test_loop


# Custom transform to ensure single-channel grayscale
class EnsureGrayscale:
    def __call__(self, image):
        if image.mode != 'L':  # Check if it's not already grayscale
            image = image.convert('L')
        return transforms.ToTensor()(image).unsqueeze(0)  # Add channel dimension (1, H, W)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def train(rank, world_size, params, output_dir):
    try:
        setup(rank, world_size)

        # For reproducibility
        set_random_seed(42)

        num_epochs = params['num_epochs']
        batch_size = params['batch_size']
        num_classes = params['num_classes']
        num_workers = params['num_workers']

        # Map the rank to the correct GPU (2 or 3)
        device = torch.device(f'cuda:{rank}')

        # Initialize the model and move it to the current device
        model = RTFNet(n_class=num_classes,
                       num_resnet_layers=params['num_resnet_layers'],
                       num_lstm_layers=params['num_lstm_layers'],
                       lstm_hidden_size=params['lstm_hidden_size'],
                       attention_heads=params['attention_heads'],
                       attention_dim=params['attention_dim'],
                       device=device).to(device)

        # Wrap the model with DDP
        model = DDP(model, device_ids=[rank])

        # Criterion and Optimizer
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=params['learning_rate'],
            momentum=params['momentum'],
            weight_decay=params['weight_decay'],
            dampening=params['dampening'],
            nesterov=params['nesterov'],
        )

        # RGB transform (for example, resizing and normalizing)
        rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Thermal transform (convert to grayscale and then to tensor)
        thermal_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            EnsureGrayscale(),  # Ensure single channel
        ])

        print ("start define dataset")

        # Create datasets and Distributed Sampler
        data_dir = params['data_dir']
        train_dataset = RGBThermalDataset(data_dir=data_dir, split='train', rgb_transform=rgb_transform, thermal_transform=thermal_transform)
        val_dataset = RGBThermalDataset(data_dir=data_dir, split='val', rgb_transform=rgb_transform, thermal_transform=thermal_transform)
        test_dataset = RGBThermalDataset(data_dir=data_dir, split='test', rgb_transform=rgb_transform, thermal_transform=thermal_transform)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        print ("finish define dataset")
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                  collate_fn=collate_fn, pin_memory=True, sampler=train_sampler, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                                collate_fn=collate_fn, pin_memory=True, sampler=val_sampler, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=True, sampler=test_sampler, persistent_workers=True)

        print ("finish loading dataset")

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_checkpoint_path = None

        for epoch in range(num_epochs):
            new_lr = get_lr_at_epoch(epoch, num_epochs)
            set_lr(optimizer, new_lr)
            start_time = time.time()
            train_sampler.set_epoch(epoch)  # Ensure all samples are used equally across all epochs
            model.train()

            # Training loop
            train_loss = 0.0
            num_batches = len(train_loader)
            for batch_idx, (rgb_images, thermal_images, labels, rgb_dirs, thermal_dirs) in enumerate(train_loader):
                optimizer.zero_grad()
                rgb_images = rgb_images.to(device)
                thermal_images = thermal_images.to(device)
                labels = labels.to(device)
                outputs, _, _, _ = model(rgb_images, thermal_images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                x = loss.item()
                train_loss += x
                # if num_batches > 100 and (batch_idx + 1) % 100 == 0:
                print(f'Rank {rank}, Batch {batch_idx + 1}/{num_batches}, Train Loss: {x}')

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch_idex, (rgb_images, thermal_images, labels, rgb_dirs, thermal_dirs) in enumerate(val_loader):
                    rgb_images = rgb_images.to(device)
                    thermal_images = thermal_images.to(device)
                    labels = labels.to(device)
                    outputs, _, _, _ = model(rgb_images, thermal_images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            epoch_time = (time.time() - start_time) / 60
            print(f"Rank {rank}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Epoch Time: {epoch_time:.2f} minutes, Learning rate: {optimizer.param_groups[0]['lr']}", flush=True)

            # Save the best model checkpoint after each epoch, Save only from rank 0 to avoid multiple saves
            if rank == 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if best_checkpoint_path is not None and os.path.exists(best_checkpoint_path):
                        os.remove(best_checkpoint_path)
                    best_checkpoint_path = os.path.join(output_dir, f"best_checkpoint_epoch_{epoch + 1}.pth.tar")
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                    }, filename=best_checkpoint_path)

        # Save final checkpoint after training finishes
        if rank == 0:
            final_checkpoint_path = os.path.join(output_dir, f"final_checkpoint.pth.tar")
            final_checkpoint = {
                'epoch': num_epochs,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }
            save_checkpoint(final_checkpoint, filename=final_checkpoint_path)

        test_loop(model, test_loader, device, rank, world_size, output_dir, load_checkpoint=False)
    finally:
        # Cleanup
        dist.destroy_process_group()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"  # Use only GPU 3
    params = {
        'num_workers': 8,
        'num_resnet_layers': 50,
        'num_lstm_layers': 1,  # can be grid searched [1,2] trying
        'lstm_hidden_size': 1024,  # can be grid searched [256, 512, 768, 1024] 1024 can fit with batch_size=5
        'num_epochs': 60,
        'batch_size': 4,   # when batch_size=3, resize can not be removed  # batch_size=5 when there is resize
        'learning_rate': 0.005,  # can be grid searched [0.00001, 0.000001]
        'data_dir': "/ssd5/meixi/ntu_data/",
        'early_stop_patience': 5,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'dampening': 0,
        'nesterov': True,
        'attention_heads': 8,
        'attention_dim': 256,
        'num_classes': 60,
    }
    world_size = 3  # Only use GPUs 1, 2 and 3
    output_dir = "/ssd5/meixi/output/60_epoch"
    mp.spawn(train, args=(world_size, params, output_dir), nprocs=world_size, join=True, start_method="spawn")
    # Delay exit to allow cleanup
    time.sleep(5)
