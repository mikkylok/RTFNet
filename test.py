import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from fvcore.nn import FlopCountAnalysis

from model.RTFNet import RTFNet
from util.RGBTDataset import RGBThermalDataset
from util.tools import setup, set_random_seed, collate_fn, find_best_checkpoint, plot_confusion_matrix


def test(rank, world_size, params, pid, output_dir):
    # Setup for distributed computing
    setup(rank, world_size)

    # For reproducibility
    set_random_seed(42)

    # Parameters
    batch_size = params['batch_size']
    num_classes = 3
    num_workers = params['num_workers']

    # Map the rank to the correct GPU (2 or 3)
    device = torch.device(f'cuda:{rank}')

    # Initialize the model and move it to the current device
    model = RTFNet(n_class=num_classes,
                   num_resnet_layers=params['num_resnet_layers'],
                   num_lstm_layers=params['num_lstm_layers'],
                   lstm_hidden_size=params['lstm_hidden_size'],
                   device=device).to(device)

    # Wrap the model with DDP
    model = DDP(model, device_ids=[rank])

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create dataset, distributed sampler and data loader
    data_dir = params['data_dir']
    test_dataset = RGBThermalDataset(data_dir=data_dir, pid=pid, split='test', transform=transform)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             collate_fn=collate_fn, pin_memory=True, sampler=test_sampler)
    test_loop(model, test_loader, device, pid, rank, world_size, output_dir, load_checkpoint=True)


def test_loop(model, test_loader, device, pid, rank, world_size, output_dir, load_checkpoint=True):
    if load_checkpoint:
        # Load the best checkpoint
        best_checkpoint_path = find_best_checkpoint(output_dir, pid)
        if best_checkpoint_path:
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            print(
                f"Loaded best checkpoint from {best_checkpoint_path}", flush=True)
        else:
            print(f"Best checkpoint not found for P{pid}, using the final model.", flush=True)

    # Testing loop
    model.eval()
    all_labels = []
    all_preds = []
    results = []
    with torch.no_grad():
        for batch_idx, (rgb_images, thermal_images, labels, lengths, rgb_dirs, thermal_dirs) in enumerate(test_loader):
            rgb_images = rgb_images.to(device)
            thermal_images = thermal_images.to(device)
            labels = labels.to(device)

            # Calculate FLOPs for the first batch
            if rank == 0 and batch_idx == 0:
                flops = FlopCountAnalysis(model, (rgb_images, thermal_images, lengths))
                tflops = flops.total() / 1e12  # Convert FLOPs to TFLOPs
                print(f"Estimated TFLOPs for a single forward pass: {tflops:.6f} TFLOPs")

            outputs, _, _ = model(rgb_images, thermal_images, lengths)
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
                }
                results.append(result)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Save to individual csvs
    df = pd.DataFrame(results)
    csv_file_path = os.path.join(output_dir, f"P{pid}_test_results_{rank}.csv")
    df.to_csv(csv_file_path, index=False)
    print(f"Saved test results to {csv_file_path}", flush=True)

    # Combine results on rank 0
    if rank == 0:
        # Concatenate all the results into one
        combined_df = []
        for i in range(world_size):
            csv_file_path = os.path.join(output_dir, f"P{pid}_test_results_{i}.csv")
            if os.path.exists(csv_file_path):
                df = pd.read_csv(csv_file_path)
                combined_df.append(df)
                os.remove(csv_file_path)
        combined_df = pd.concat(combined_df, ignore_index=True)

        # Calculate test metrics
        true_labels = combined_df['true_label']
        predictions = combined_df['prediction']
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
        print(f"Test Results for P{pid}: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        # Save confusion matrix
        class_names = ['Negative', 'Smoking', 'Eating']
        plot_confusion_matrix(true_labels, predictions, class_names, pid, output_dir)

        # Save the combined DataFrame to the final CSV
        final_csv_path = os.path.join(output_dir, f"P{pid}_test_results.csv")
        combined_df.to_csv(final_csv_path, index=False)  # Save the combined results
        print(f"Saved combined test results to {final_csv_path}", flush=True)

    # Cleanup
    dist.destroy_process_group()


def lopo_test(params, world_size, participant_pids, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for pid in participant_pids:
        mp.spawn(test, args=(world_size, params, pid, output_dir), nprocs=world_size, join=True)


if __name__ == '__main__':
    params = {
        'num_workers': 16,
        'num_resnet_layers': 50,
        'num_lstm_layers': 1,
        'lstm_hidden_size': 1024,
        'batch_size': 5,
        'data_dir': "/home/meixi/data",
    }
    world_size = 3  # Use GPUs 1, 2, and 3
    output_dir = "/home/meixi/mid_fusion/rtfnet/output"
    participant_pids = [6, 7, 13, 14, 15, 16, 18]

    # Call the test function for each participant
    lopo_test(params, world_size, participant_pids, output_dir)
