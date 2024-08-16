import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd


class RGBThermalDataset(Dataset):
    def __init__(self, data_dir, pid, split, input_h=480, input_w=640, transform=None):
        super(RGBThermalDataset, self).__init__()

        self.image_dir = os.path.join(data_dir, f'P{pid}', 'rgbt-mid-fusion-rtfnet', 'image')
        self.labels_path = os.path.join(data_dir, f'P{pid}', 'rgbt-mid-fusion-rtfnet', 'label', f'{split}.csv')

        # Load labels
        self.labels_df = pd.read_csv(self.labels_path)

        # Other attributes
        self.input_h = input_h
        self.input_w = input_w
        self.transform = transform  # Add transform attribute

    def read_image(self, image_dir, image_type):
        image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
        images = []
        for image_path in image_paths:
            try:
                image = Image.open(image_path).resize((self.input_w, self.input_h))
                if self.transform:
                    image = self.transform(image)  # Apply the transform
                else:
                    image = image.resize((self.input_w, self.input_h))
                    image = np.asarray(image, dtype=np.float32) / 255.0
                    if image_type == 'thermal':
                        image = np.expand_dims(image, axis=-1)  # Ensure thermal is (H, W, 1)
                    image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
                    image = torch.from_numpy(image)
                images.append(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
        return torch.stack(images)  # Stack to get shape (num_frames, C, H, W)

    def __getitem__(self, index):
        # Get the timestamp and label
        timestamp, label = self.labels_df.iloc[index]

        # Define the image directories for RGB and thermal images
        rgb_dir = os.path.join(self.image_dir, str(timestamp), 'rgb')
        thermal_dir = os.path.join(self.image_dir, str(timestamp), 'thermal')

        # Read and sort images
        rgb_images = self.read_image(rgb_dir, 'rgb')
        thermal_images = self.read_image(thermal_dir, 'thermal')

        return rgb_images, thermal_images, torch.tensor(label), str(timestamp)

    def __len__(self):
        return len(self.labels_df)
