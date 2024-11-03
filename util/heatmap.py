import os
import cv2
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, rgb_image, thermal_image, target_class):
        output, _, _, _ = self.model(rgb_image.unsqueeze(0), thermal_image.unsqueeze(0))
        loss = output[0, target_class]

        # Backward pass
        self.model.zero_grad()
        loss.backward()

        # Compute Grad-CAM heatmap
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(len(pooled_gradients)):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.clamp(heatmap, min=0)
        heatmap /= torch.max(heatmap)
        return heatmap


class TemporalHeatmap:
    def __init__(self, lstm):
        self.lstm = lstm

    def compute_hidden_state_norms(self, lstm_outputs):
        # lstm_outputs shape: (batch_size, seq_length, hidden_size)
        hidden_state_norms = torch.norm(lstm_outputs, dim=2)  # L2 norm along the hidden size
        return hidden_state_norms


def overlay_heatmap(heatmap, img):
    img = cv2.resize(img, (224, 224))
    heatmap = 1.0 - heatmap.detach().cpu().numpy()
    # heatmap = heatmap.detach().cpu().numpy()
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_VIRIDIS)

    if img.max() <= 1:
        img = np.uint8(255 * img)  # If image is in [0, 1] range, scale it up to [0, 255]
    else:
        img = np.uint8(img)

    overlayed_img = cv2.addWeighted(heatmap_colored, 0.4, img, 0.6, 0)
    return overlayed_img


def save_spatial_heatmap(heatmap, path, frame_index, modality='rgb'):
    """
    Save the spatial heatmap overlay to a file.

    Parameters:
    - heatmap: The overlayed heatmap image (numpy array)
    - path: Directory to save the heatmap image
    - frame_index: Frame number (for naming)
    - modality: Either 'rgb' or 'thermal' to indicate modality
    """
    filename = f"{modality}_heatmap_frame_{frame_index}.png"
    cv2.imwrite(os.path.join(path, filename), heatmap)


def save_temporal_heatmap(temporal_heatmap, path, mode="heatmap"):
    """
    Save the temporal heatmap (LSTM hidden state norm heatmap) to a file with smooth transitions.

    Parameters:
    - temporal_heatmap: LSTM hidden state norms over time
    - path: Directory to save the temporal heatmap
    - mode: line, grid_heatmap, heatmap
    """
    if mode == 'line':
        plt.plot(temporal_heatmap)
        plt.title("LSTM Hidden State Norm (Temporal Heatmap)")
        plt.xlabel("Time Step (Frame Index)")
        plt.ylabel("L2 Norm of Hidden State")
        plt.savefig(os.path.join(path, "temporal_heatmap.png"))
        plt.close()
    elif mode == 'grid_heatmap':
        # Ensure temporal_heatmap is in 2D format (if it's 1D, reshape it)
        temporal_heatmap = np.reshape(temporal_heatmap, (1, -1))  # Shape to [1, time_steps] for heatmap

        # Set the plot size
        plt.figure(figsize=(12, 2))  # Adjust figure size for a long, narrow heatmap

        # Create a heatmap with the "coolwarm" colormap
        sns.heatmap(temporal_heatmap, cmap="coolwarm", cbar=True, annot=True, fmt=".2f", xticklabels=1)

        # Set the labels and title
        plt.title("LSTM Hidden State Norm (Temporal Heatmap)")
        plt.xlabel("Time Step (Frame Index)")
        plt.ylabel("Hidden State Norm")

        # Save the heatmap as a PNG file
        plt.savefig(os.path.join(path, "temporal_heatmap.png"), bbox_inches='tight', dpi=300)
        plt.close()
    elif mode == 'heatmap':
        # Ensure temporal_heatmap is in 2D format (if it's 1D, reshape it)
        temporal_heatmap = np.reshape(temporal_heatmap, (1, -1))  # Shape to [1, time_steps]

        # Set the plot size
        plt.figure(figsize=(12, 2))  # Adjust figure size for a long, narrow heatmap

        # Use imshow for smooth transitions
        plt.imshow(temporal_heatmap, aspect='auto', cmap='coolwarm', interpolation='bilinear')

        # Add colorbar for reference
        plt.colorbar()

        # Set the labels and title
        plt.title("LSTM Hidden State Norm (Temporal Heatmap)")
        plt.xlabel("Time Step (Frame Index)")
        plt.ylabel("Hidden State Norm")

        # Save the heatmap as a PNG file
        plt.savefig(os.path.join(path, "temporal_heatmap.png"), bbox_inches='tight', dpi=300)
        plt.close()


def save_rgb_thermal_weight_heatmap(rgb_weight, thermal_weight, path, time_step):
    print (rgb_weight, thermal_weight, time_step)
    rgb_weight = rgb_weight.cpu().numpy()
    thermal_weight = thermal_weight.cpu().numpy()

    # Create figure and axis
    plt.figure(figsize=(10, 2))  # Smaller height for single bar

    # Plot horizontal stacked bar
    plt.barh(time_step, rgb_weight, label='RGB Weight', color='skyblue')
    plt.barh(time_step, thermal_weight, left=rgb_weight, label='Thermal Weight', color='orange')

    # Add labels and title
    plt.ylabel('Time Step')
    plt.xlabel('Weight Contribution')
    plt.title(f'RGB vs Thermal Weight Contribution at Time Step {time_step}')
    plt.legend()

    # Save the heatmap as a PNG file
    plt.savefig(os.path.join(path, f"weight_heatmap_{time_step}.png"), bbox_inches='tight', dpi=300)
    plt.close()


def save_spatial_heatmaps_side_by_side(overlay_rgb_frames, overlay_thermal_frames, output_dir):
    """
    Save spatial heatmaps (RGB and thermal) side by side for each time step.

    Parameters:
    - overlay_rgb_frames: List of RGB overlay frames (numpy arrays).
    - overlay_thermal_frames: List of thermal overlay frames (numpy arrays).
    - output_dir: Directory to save the heatmaps.
    """
    num_frames = len(overlay_rgb_frames)

    # Create a figure with two rows: RGB in the first row and thermal in the second
    fig, axs = plt.subplots(2, num_frames, figsize=(num_frames * 5, 10))  # Adjust figure size

    # Plot RGB overlays in the first row
    for t in range(num_frames):
        axs[0, t].imshow(overlay_rgb_frames[t])
        axs[0, t].axis('off')  # Turn off axis
        axs[0, t].set_title(f"RGB Frame {t + 1}")

    # Plot thermal overlays in the second row
    for t in range(num_frames):
        axs[1, t].imshow(overlay_thermal_frames[t])
        axs[1, t].axis('off')  # Turn off axis
        axs[1, t].set_title(f"Thermal Frame {t + 1}")

    # Save the figure to the output directory
    output_path = os.path.join(output_dir, "overlay_rgb_thermal_side_by_side.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Spatial heatmaps (side by side) saved in {output_path}")
