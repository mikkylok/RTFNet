import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


def temporal_sampling(frames, num_samples):
    """
    Sample num_samples frames with equal intervals from the provided frames tensor.
    """
    start_idx = 0
    end_idx = len(frames) - 1
    if len(frames) < num_samples:
        index = torch.linspace(start_idx, end_idx, len(frames))  # Use all frames if fewer than num_samples
    else:
        index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, len(frames) - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames


class RGBThermalDataset(Dataset):
    def __init__(self, data_dir, split, rgb_transform=None, thermal_transform=None, target_num_frames=9):
        super(RGBThermalDataset, self).__init__()

        self.image_dir = os.path.join(data_dir, 'image')
        self.labels_path = os.path.join(data_dir, 'label', f'{split}.csv')

        # Load labels
        self.labels_df = pd.read_csv(self.labels_path)

        # Transforms
        self.rgb_transform = rgb_transform
        self.thermal_transform = thermal_transform
        self.target_num_frames = target_num_frames

    def read_image(self, image_dir, image_type):
        image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
        images = []
        for image_path in image_paths:
            try:
                image = Image.open(image_path)
                # Apply specific transform
                if image_type == 'rgb' and self.rgb_transform:
                    image = self.rgb_transform(image)
                elif image_type == 'thermal' and self.thermal_transform:
                    image = self.thermal_transform(image)
                images.append(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

        images = torch.stack(images)  # Shape (num_frames, C, H, W)
        images = temporal_sampling(images, self.target_num_frames)

        if image_type == 'thermal':
            images = images.squeeze(1)  # Remove any extra dimensions for thermal images

        return images

    def __getitem__(self, index):
        # Get the timestamp and label
        video_image_path, label = self.labels_df.iloc[index]

        # Define the image directories for RGB and thermal images
        rgb_dir = os.path.join(video_image_path, 'rgb')
        thermal_dir = os.path.join(video_image_path, 'ir')

        # Read and sort images
        rgb_images = self.read_image(rgb_dir, 'rgb')
        thermal_images = self.read_image(thermal_dir, 'thermal')

        return rgb_images, thermal_images, torch.tensor(label), rgb_dir, thermal_dir

    def __len__(self):
        return len(self.labels_df)