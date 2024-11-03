import os
import re
import cv2
import time
from PIL import Image

import torch
from torchvision import transforms

from model.RTFNet import RTFNet
from util.tools import setup, set_random_seed, find_best_checkpoint
from util.heatmap import GradCAM, overlay_heatmap, save_spatial_heatmap, save_temporal_heatmap, save_rgb_thermal_weight_heatmap, save_spatial_heatmaps_side_by_side


def get_timestamp_from_path(video_path):
    file_name = os.path.basename(video_path)

    # Use regular expression to extract the desired substring
    match = re.search(r'(\d+_\d+)_', file_name)

    return match.group(1)


def load_video_pair(rgb_video_path, thermal_video_path, transform):
    """
    Load RGB and thermal video frames, apply transformations, and return tensors.
    """
    rgb_frames = []
    thermal_frames = []
    original_rgb_frames = []
    original_thermal_frames = []

    # Load RGB video
    rgb_cap = cv2.VideoCapture(rgb_video_path)
    while rgb_cap.isOpened():
        ret, frame = rgb_cap.read()
        if not ret:
            break
        original_rgb_frames.append(frame)
        frame_rgb_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_rgb = transform(frame_rgb_pil).unsqueeze(0)
        rgb_frames.append(frame_rgb)

    rgb_cap.release()

    # Load thermal video
    thermal_cap = cv2.VideoCapture(thermal_video_path)
    while thermal_cap.isOpened():
        ret, frame = thermal_cap.read()
        if not ret:
            break
        original_thermal_frames.append(frame)
        frame_gray_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        frame_thermal = transform(frame_gray_pil).unsqueeze(0)
        thermal_frames.append(frame_thermal)

    thermal_cap.release()

    # Stack frames along time dimension (for batch processing)
    rgb_frames = torch.cat(rgb_frames, dim=0).unsqueeze(0)  # (1, num_frames, channels, H, W)
    thermal_frames = torch.cat(thermal_frames, dim=0).unsqueeze(0)  # (1, num_frames, 1, H, W)

    return rgb_frames, thermal_frames, original_rgb_frames, original_thermal_frames


def test(rank, params, pid, output_dir, rgb_video_path, thermal_video_path, checkpoint_dir):
    set_random_seed(42)
    device = torch.device(f'cuda:{rank}')

    # Initialize the model and move it to the current device
    model = RTFNet(n_class=3,
                   num_resnet_layers=params['num_resnet_layers'],
                   num_lstm_layers=params['num_lstm_layers'],
                   lstm_hidden_size=params['lstm_hidden_size'],
                   device=device,
                   attention_heads=params['attention_heads'],
                   attention_dim=params['attention_dim']).to(device)

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load RGB and thermal videos
    rgb_frames, thermal_frames, original_rgb_frames, original_thermal_frames = load_video_pair(rgb_video_path, thermal_video_path, transform)

    timestamp = get_timestamp_from_path(rgb_video_path)
    output_dir = os.path.join(output_dir, timestamp)

    # Run test loop for inference
    test_loop(model, rgb_frames, thermal_frames, original_rgb_frames, original_thermal_frames, device, pid, output_dir, checkpoint_dir)


# def test_loop(model, rgb_images, thermal_images, original_rgb_frames, original_thermal_frames, device, pid, output_dir, checkpoint_dir):
#     # Load the best checkpoint
#     best_checkpoint_path = find_best_checkpoint(checkpoint_dir, pid)
#     if best_checkpoint_path:
#         checkpoint = torch.load(best_checkpoint_path, map_location=device)
#
#         # Remove 'module.' prefix from keys if the model was saved with DataParallel or DDP
#         state_dict = checkpoint['state_dict']
#         new_state_dict = {}
#         for key, value in state_dict.items():
#             new_key = key.replace('module.', '')  # Remove the 'module.' prefix
#             new_state_dict[new_key] = value
#
#         model.load_state_dict(new_state_dict)
#         print(
#             f"Loaded best checkpoint from {best_checkpoint_path}", flush=True)
#     else:
#         print(f"Best checkpoint not found for P{pid}, using the final model.", flush=True)
#
#     # Testing loop
#     model.eval()
#     start_time = time.time()
#     torch.cuda.reset_peak_memory_stats(device)
#     with torch.no_grad():
#         rgb_images = rgb_images.to(device)
#         thermal_images = thermal_images.to(device)
#         outputs, hidden_states, rgb_weights, thermal_weights = model(rgb_images, thermal_images)
#
#     # Print inference time
#     end_time = time.time()
#     inference_time = end_time - start_time
#     print(f"Inference time for video pair: {inference_time:.4f} seconds")
#
#     # Get GPU memory usage
#     current_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Convert to MB
#     peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
#     print(f"Current GPU Memory Usage: {current_memory:.2f} MB")
#     print(f"Peak GPU Memory Usage: {peak_memory:.2f} MB")
#
#     # Get the predicted label
#     probs = torch.softmax(outputs, dim=1)
#     _, preds = torch.max(outputs, 1)
#     predicted_label = preds.cpu().item()
#     print(f"Predicted label: {predicted_label}, Probabilities: {probs.cpu().numpy()}")
#
#     # Grad-CAM integration for spatial heatmaps
#     grad_cam_rgb = GradCAM(model, model.encoder_rgb_layer4)  # Register Grad-CAM for RGB
#     grad_cam_thermal = GradCAM(model, model.encoder_thermal_layer4)  # Register Grad-CAM for Thermal
#
#     spatial_output_dir = os.path.join(output_dir, "spatial_heatmaps")
#     os.makedirs(spatial_output_dir, exist_ok=True)
#
#     weight_output_dir = os.path.join(output_dir, "weight_heatmaps")
#     os.makedirs(weight_output_dir, exist_ok=True)
#
#     for t in range(rgb_images.shape[1]):  # Iterate over time steps
#         rgb_frame = rgb_images[:, t, :, :, :]
#         thermal_frame = thermal_images[:, t, :, :, :]
#
#         # Temporarily switch model to training mode for Grad-CAM backprop
#         model.train()
#
#         # Generate heatmaps
#         heatmap_rgb = grad_cam_rgb(rgb_frame, thermal_frame, predicted_label)
#         heatmap_thermal = grad_cam_thermal(rgb_frame, thermal_frame, predicted_label)
#
#         # Switch back to evaluation mode after Grad-CAM
#         model.eval()
#
#         # Overlay heatmaps on original frames and save
#         overlay_rgb = overlay_heatmap(heatmap_rgb, original_rgb_frames[t])
#         overlay_thermal = overlay_heatmap(heatmap_thermal, original_thermal_frames[t])
#
#         save_spatial_heatmap(overlay_rgb, spatial_output_dir, t, modality='rgb')
#         save_spatial_heatmap(overlay_thermal, spatial_output_dir, t, modality='thermal')
#
#         save_rgb_thermal_weight_heatmap(rgb_weights[t], thermal_weights[t], weight_output_dir, t)
#
#     print(f"Spatial heatmaps saved in {spatial_output_dir}")
#
#     # Temporal heatmap using LSTM hidden states
#     hidden_state_norms = torch.norm(hidden_states, dim=2).cpu().numpy()  # Compute L2 norm of hidden states
#     temporal_output_dir = os.path.join(output_dir, "temporal_heatmaps")
#     os.makedirs(temporal_output_dir, exist_ok=True)
#
#     save_temporal_heatmap(hidden_state_norms[0], temporal_output_dir)
#     print(f"Temporal heatmap saved in {temporal_output_dir}")


def test_loop(model, rgb_images, thermal_images, original_rgb_frames, original_thermal_frames, device, pid, output_dir, checkpoint_dir):
    # Load the best checkpoint
    best_checkpoint_path = find_best_checkpoint(checkpoint_dir, pid)
    if best_checkpoint_path:
        checkpoint = torch.load(best_checkpoint_path, map_location=device)

        # Remove 'module.' prefix from keys if the model was saved with DataParallel or DDP
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '')  # Remove the 'module.' prefix
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict)
        print(
            f"Loaded best checkpoint from {best_checkpoint_path}", flush=True)
    else:
        print(f"Best checkpoint not found for P{pid}, using the final model.", flush=True)

    # Testing loop
    model.eval()
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        rgb_images = rgb_images.to(device)
        thermal_images = thermal_images.to(device)
        outputs, hidden_states, rgb_weights, thermal_weights = model(rgb_images, thermal_images)

    # Print inference time
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time for video pair: {inference_time:.4f} seconds")

    # Get GPU memory usage
    current_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Convert to MB
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
    print(f"Current GPU Memory Usage: {current_memory:.2f} MB")
    print(f"Peak GPU Memory Usage: {peak_memory:.2f} MB")

    # Get the predicted label
    probs = torch.softmax(outputs, dim=1)
    _, preds = torch.max(outputs, 1)
    predicted_label = preds.cpu().item()
    print(f"Predicted label: {predicted_label}, Probabilities: {probs.cpu().numpy()}")

    # Grad-CAM integration for spatial heatmaps
    grad_cam_rgb = GradCAM(model, model.encoder_rgb_layer4)  # Register Grad-CAM for RGB
    grad_cam_thermal = GradCAM(model, model.encoder_thermal_layer4)  # Register Grad-CAM for Thermal

    overlay_rgb_frames = []
    overlay_thermal_frames = []

    for t in range(rgb_images.shape[1]):  # Iterate over time steps
        rgb_frame = rgb_images[:, t, :, :, :]
        thermal_frame = thermal_images[:, t, :, :, :]

        # Temporarily switch model to training mode for Grad-CAM backprop
        model.train()

        # Generate heatmaps
        heatmap_rgb = grad_cam_rgb(rgb_frame, thermal_frame, predicted_label)
        heatmap_thermal = grad_cam_thermal(rgb_frame, thermal_frame, predicted_label)

        # Switch back to evaluation mode after Grad-CAM
        model.eval()

        # Overlay heatmaps on original frames and save
        overlay_rgb = overlay_heatmap(heatmap_rgb, original_rgb_frames[t])
        overlay_thermal = overlay_heatmap(heatmap_thermal, original_thermal_frames[t])

        overlay_rgb_frames.append(overlay_rgb)
        overlay_thermal_frames.append(overlay_thermal)

    # Save the side-by-side spatial heatmaps
    spatial_output_dir = os.path.join(output_dir, "spatial_heatmaps_combined")
    os.makedirs(spatial_output_dir, exist_ok=True)
    save_spatial_heatmaps_side_by_side(overlay_rgb_frames, overlay_thermal_frames, spatial_output_dir)



if __name__ == '__main__':
    params = {
        'num_resnet_layers': 50,
        'num_lstm_layers': 1,
        'lstm_hidden_size': 1024,
        'attention_heads': 8,
        'attention_dim': 256,
    }
    test(rank=0,
         params=params,
         pid=7,
         output_dir="/home/meixi/mid_fusion/rtfnet/output/heatmap",
         rgb_video_path="/ssd2/R21_Clips/P7/eating-rgb_resized/1677760499000_1677760500800_10.mp4",
         thermal_video_path="/ssd2/R21_Clips/P7/eating-thermal_resized/1677760499000_1677760500800_10.mp4",
         checkpoint_dir="/home/meixi/mid_fusion/rtfnet/output/early_cross_attention_late_fusion_no_skip_connection_8_256")
    # test(rank=0,
    #      params=params,
    #      pid=6,
    #      output_dir="/home/meixi/mid_fusion/rtfnet/output/heatmap",
    #      rgb_video_path="/ssd2/R21_Clips/P6/eating-rgb_resized/1678201636200_1678201637200_5.mp4",
    #      thermal_video_path="/ssd2/R21_Clips/P6/eating-thermal_resized/1678201636200_1678201637200_5.mp4",
    #      checkpoint_dir="/home/meixi/mid_fusion/rtfnet/output/early_cross_attention_late_fusion_no_skip_connection_8_256")