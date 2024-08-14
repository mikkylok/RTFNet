import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd


class RGBThermalDataset(Dataset):
    def __init__(self, data_dir, pid, split, input_h=480, input_w=640):
        super(RGBThermalDataset, self).__init__()

        self.rgb_path = os.path.join(data_dir, f'P{pid}', 'rgbt-mid-fusion-rtfnet', 'image')
        self.thermal_path = os.path.join(data_dir, f'P{pid}', 'rgbt-mid-fusion-rtfnet', 'image')
        self.labels_path = os.path.join(data_dir, f'P{pid}', 'rgbt-mid-fusion-rtfnet', 'label', f'{split}.csv')

        # Load labels
        self.labels_df = pd.read_csv(self.labels_path)

        # Other attributes
        self.input_h = input_h
        self.input_w = input_w

    def read_image(self, timestamp, frame_id, image_type, image_format='png'):
        folder = 'rgb' if image_type == 'rgb' else 'thermal'
        image_path = os.path.join(self.rgb_path if image_type == 'rgb' else self.thermal_path,
                                  str(timestamp),
                                  folder,
                                  f'{timestamp}_{frame_id}.{image_format}')
        image = Image.open(image_path).resize((self.input_w, self.input_h))
        image = np.asarray(image, dtype=np.float32) / 255.0

        if image_type == 'thermal':
            image = np.expand_dims(image, axis=-1)  # Make sure thermal is (H, W, 1)

        image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
        return torch.from_numpy(image)

    def __getitem__(self, index):
        # Get the timestamp and label
        timestamp, label = self.labels_df.iloc[index]

        # Initialize lists to hold the sequence of images
        rgb_images = []
        thermal_images = []

        # Assume the sequence length is determined by the number of frames available for the timestamp
        frame_id = 1
        while True:
            try:
                rgb_image = self.read_image(timestamp, frame_id, 'rgb')
                thermal_image = self.read_image(timestamp, frame_id, 'thermal')
                rgb_images.append(torch.tensor(rgb_image))
                thermal_images.append(torch.tensor(thermal_image))
                frame_id += 1
            except FileNotFoundError:
                break

        # Convert lists to tensors with shape (num_frames, C, H, W)
        rgb_images = torch.stack(rgb_images)
        thermal_images = torch.stack(thermal_images)

        return rgb_images, thermal_images, torch.tensor(label), str(timestamp)

    def __len__(self):
        return len(self.labels_df)

