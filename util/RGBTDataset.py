import os
import cv2
import math
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


def temporal_sampling(image_paths, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal intervals. If num_samples is greater than
    the number of frames, the frames are evenly repeated.
    Args:
        image_paths (list): a list of image paths.
        num_samples (int): number of frames to sample.
    Returns:
        sampled_paths (list): a list of sampled image paths.
    """
    num_frames = len(image_paths)

    if num_frames >= num_samples:
        # If there are more frames than samples, do normal sampling
        indices = np.linspace(0, num_frames - 1, num_samples).astype(int)
    else:
        # If there are fewer frames than samples, repeat frames evenly
        repeat_factors = np.ceil(np.linspace(0, num_samples, num_frames + 1)).astype(int)
        indices = np.hstack([np.full((repeat_factors[i + 1] - repeat_factors[i],), i) for i in range(num_frames)])

    sampled_paths = [image_paths[i] for i in indices]

    return sampled_paths


def random_short_side_scale_jitter(images, min_size, max_size, size=None):
    """
    Perform a spatial short scale jittering on the given images and
    corresponding boxes using NumPy.
    Args:
        images (numpy array): images to perform scale jitter. Dimension is
            `num frames` x `height` x `width` x `channel`.
        min_size (int): the minimal size to scale the frames.
        max_size (int): the maximal size to scale the frames.
    Returns:
        (numpy array): the scaled images with dimension of
            `num frames` x `new height` x `new width` x `channel`.
    """
    if size is None:
        size = int(round(np.random.uniform(min_size, max_size)))

    height, width = images.shape[1], images.shape[2]
    if (width <= height and width == size) or (
            height <= width and height == size
    ):
        return images
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
        new_width = size
    else:
        new_width = int(math.floor((float(width) / height) * size))
        new_height = size

    resized_images = np.array([cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                               for image in images])

    return resized_images, size


def random_crop(images, size, x_offset=None, y_offset=None):
    """
    Perform random spatial crop on the given images.
    Args:
        images (numpy array): images to perform random crop. The dimension is
            `num frames` x `height` x `width` x `channel`.
        size (int): the size of height and width to crop on the image.
    Returns:
        cropped (numpy array): cropped images with dimension of
            `num frames` x `size` x `size` x `channel`.
    """
    height, width = images.shape[1], images.shape[2]

    if x_offset is None and y_offset is None:
        y_offset = np.random.randint(0, height - size) if height > size else 0
        x_offset = np.random.randint(0, width - size) if width > size else 0

    cropped = images[:, y_offset:y_offset + size, x_offset:x_offset + size, :]

    return cropped, x_offset, y_offset


def horizontal_flip(prob, images, random_prob=None):
    """
    Perform horizontal flip on the given images.
    Args:
        prob (float): probability to flip the images.
        images (numpy array): images to perform horizontal flip, the dimension is
            `num frames` x `height` x `width` x `channel`.
    Returns:
        images (numpy array): images with dimension of
            `num frames` x `height` x `width` x `channel`.
    """
    if random_prob is None:
        random_prob = np.random.uniform()
    if random_prob < prob:
        images = np.flip(images, axis=2)  # Flip along the width dimension
    return images, random_prob


def spatial_sampling(
    frames,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    jitter_size=None,
    crop_x_offset=None,
    crop_y_offset=None,
    random_filp_prob=None
):
    """
    Perform spatial sampling on the given video frames using NumPy arrays.
    Args:
        frames (numpy array): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
    Returns:
        frames (numpy array): spatially sampled frames.
    """
    frames, size = random_short_side_scale_jitter(frames, min_scale, max_scale, jitter_size)
    frames, x_offset, y_offset = random_crop(frames, crop_size, crop_x_offset, crop_y_offset)
    frames, random_prob = horizontal_flip(0.5, frames, random_filp_prob)
    return frames, size, x_offset, y_offset, random_prob


def normalize(array, mean, std):
    """
    Normalize a given NumPy array by subtracting the mean and dividing by the std.
    Args:
        array (numpy array): array to normalize.
        mean (numpy array or list): mean value to subtract.
        std (numpy array or list): std to divide.
    """
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0

    if isinstance(mean, list):
        mean = np.array(mean).reshape(1, 1, 1, -1)  # Reshape to (1, 1, 1, C)
    if isinstance(std, list):
        std = np.array(std).reshape(1, 1, 1, -1)    # Reshape to (1, 1, 1, C)

    array = (array - mean) / std

    return array


class RGBThermalDataset(Dataset):
    def __init__(self, data_dir, pid, split, input_h=480, input_w=640, transform=None, target_num_frames=9):
        super(RGBThermalDataset, self).__init__()

        self.image_dir = os.path.join(data_dir, f'P{pid}', 'rgbt-mid-fusion-rtfnet', 'image')
        self.labels_path = os.path.join(data_dir, f'P{pid}', 'rgbt-mid-fusion-rtfnet', 'label', f'{split}.csv')

        # Load labels
        self.labels_df = pd.read_csv(self.labels_path)

        # Other attributes
        self.input_h = input_h
        self.input_w = input_w
        self.transform = transform
        self.target_num_frames = target_num_frames  # Target number of frames

    def read_image(self, image_dir, image_type):
        image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
        # Apply temporal sampling to ensure all sequences have the same number of frames
        image_paths = temporal_sampling(image_paths, self.target_num_frames)

        images = []
        for image_path in image_paths:
            image = Image.open(image_path)
            if self.transform == 'resize':
                image = image.resize((self.input_w, self.input_h))
            image = np.asarray(image)
            if image_type == 'thermal':
                image = np.expand_dims(image, axis=-1)  # Ensure thermal is (H, W, 1)
            images.append(image)

        images = np.stack(images)  # Stack to get shape (num_frames, H, W, C) without converting to tensor

        return images

    def __getitem__(self, index):
        # Get the timestamp and label
        timestamp, label = self.labels_df.iloc[index]

        # Define the image directories for RGB and thermal images
        rgb_dir = os.path.join(self.image_dir, str(timestamp), 'rgb')
        thermal_dir = os.path.join(self.image_dir, str(timestamp), 'thermal')

        # Read and sort images
        rgb_images = self.read_image(rgb_dir, 'rgb')
        thermal_images = self.read_image(thermal_dir, 'thermal')

        if self.transform == 'crop':
            # Perform data augmentation
            rgb_images, size, x_offset, y_offset, random_flip_prob = spatial_sampling(rgb_images)
            thermal_images, _, _, _, _ = spatial_sampling(thermal_images,
                                              jitter_size=size,
                                              crop_x_offset=x_offset,
                                              crop_y_offset=y_offset,
                                              random_filp_prob=random_flip_prob)

        # Perform color normalization.
        rgb_images = normalize(rgb_images, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        thermal_images = normalize(thermal_images, [0.5], [0.5])

        # Convert to PyTorch tensors
        rgb_images = torch.from_numpy(rgb_images).float()
        thermal_images = torch.from_numpy(thermal_images).float()

        # Rearrange dimensions from (H, W, C) to (C, H, W)
        rgb_images = rgb_images.permute(0, 3, 1, 2)  # Assuming rgb_images has shape (num_frames, H, W, C)
        thermal_images = thermal_images.permute(0, 3, 1, 2)  # Assuming thermal_images has shape (num_frames, H, W, C)

        return rgb_images, thermal_images, torch.tensor(label), str(timestamp)

    def __len__(self):
        return len(self.labels_df)
