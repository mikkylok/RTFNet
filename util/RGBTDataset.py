import os
import math
import numpy as np
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


def crop_boxes(boxes, x_offset, y_offset):
    """
    Peform crop on the bounding boxes given the offsets.
    Args:
        boxes (ndarray or None): bounding boxes to peform crop. The dimension
            is `num boxes` x 4.
        x_offset (int): cropping offset in the x axis.
        y_offset (int): cropping offset in the y axis.
    Returns:
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

    return cropped_boxes


def random_short_side_scale_jitter(images, min_size, max_size, size=None):
    """
    Perform a spatial short scale jittering on the given images and
    corresponding boxes.
    Args:
        images (tensor): images to perform scale jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
        min_size (int): the minimal size to scale the frames.
        max_size (int): the maximal size to scale the frames.
    Returns:
        (tensor): the scaled images with dimension of
            `num frames` x `channel` x `new height` x `new width`.
    """
    if size is None:
        size = int(round(np.random.uniform(min_size, max_size)))

    height = images.shape[2]
    width = images.shape[3]
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return images
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))

    return (
        torch.nn.functional.interpolate(
            images,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        ), size
    )


def random_crop(images, size, x_offset=None, y_offset=None):
    """
    Perform random spatial crop on the given images and corresponding boxes.
    Args:
        images (tensor): images to perform random crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): the size of height and width to crop on the image.
    Returns:
        cropped (tensor): cropped images with dimension of
            `num frames` x `channel` x `size` x `size`.
    """
    if images.shape[2] == size and images.shape[3] == size:
        return images, None
    height = images.shape[2]
    width = images.shape[3]

    if x_offset is None and y_offset is None:
        y_offset = 0
        if height > size:
            y_offset = int(np.random.randint(0, height - size))
        x_offset = 0
        if width > size:
            x_offset = int(np.random.randint(0, width - size))

    cropped = images[
        :, :, y_offset : y_offset + size, x_offset : x_offset + size
    ]

    return cropped, x_offset, y_offset


def horizontal_flip(prob, images, random_prob=None):
    """
    Perform horizontal flip on the given images and corresponding boxes.
    Args:
        prob (float): probility to flip the images.
        images (tensor): images to perform horizontal flip, the dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): images with dimension of
            `num frames` x `channel` x `height` x `width`.
    """
    if random_prob is None:
        random_prob = np.random.uniform()
    if random_prob < prob:
        images = images.flip((-1))
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
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
    Returns:
        frames (tensor): spatially sampled frames.
    """
    frames, size = random_short_side_scale_jitter(frames, min_scale, max_scale, jitter_size)
    frames, x_offset, y_offset = random_crop(frames, crop_size, crop_x_offset, crop_y_offset)
    frames, random_prob = horizontal_flip(0.5, frames, random_filp_prob)
    return frames, size, x_offset, y_offset, random_prob


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean).view(-1, 1, 1)
    if type(std) == list:
        std = torch.tensor(std).view(-1, 1, 1)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


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
        self.transform = transform  # Add transform attribute
        self.target_num_frames = target_num_frames  # Target number of frames

    def read_image(self, image_dir, image_type):
        image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
        images = []
        for image_path in image_paths:
            try:
                image = Image.open(image_path).resize((self.input_w, self.input_h))
                if self.transform:
                    image = self.transform(image)  # Apply the transform
                else:
                    image = np.asarray(image, dtype=np.float32) / 255.0
                    if image_type == 'thermal':
                        image = np.expand_dims(image, axis=-1)  # Ensure thermal is (H, W, 1)
                    image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
                    image = torch.from_numpy(image)
                images.append(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

        images = torch.stack(images)  # Stack to get shape (num_frames, C, H, W)

        # Apply temporal sampling to ensure all sequences have the same number of frames
        images = temporal_sampling(images, self.target_num_frames)

        # Perform color normalization.
        if image_type == 'rgb':
            images = tensor_normalize(images, [0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
        else:
            images = tensor_normalize(images, [0.5], [0.5])  # For single-channel images

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

        # Perform data augmentation.
        rgb_images, size, x_offset, y_offset, random_flip_prob = spatial_sampling(rgb_images)
        thermal_images, _, _, _, _ = spatial_sampling(thermal_images,
                                          jitter_size=size,
                                          crop_x_offset=x_offset,
                                          crop_y_offset=y_offset,
                                          random_filp_prob=random_flip_prob)

        return rgb_images, thermal_images, torch.tensor(label), str(timestamp)

    def __len__(self):
        return len(self.labels_df)
