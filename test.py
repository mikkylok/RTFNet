import cv2
import numpy as np
import torch
from model.RTFNet import RTFNet


def process_video(rgb_video_path, thermal_video_path):
    # Open the video files
    rgb_cap = cv2.VideoCapture(rgb_video_path)
    thermal_cap = cv2.VideoCapture(thermal_video_path)

    # Ensure both videos are open
    if not rgb_cap.isOpened() or not thermal_cap.isOpened():
        print("Error: Could not open video files.")
        return

    # Get the total number of frames (assuming both videos have the same number of frames)
    num_frames = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize lists to hold frames
    rgb_frames = []
    thermal_frames = []

    # Read and process each frame
    for _ in range(num_frames):
        ret_rgb, rgb_frame = rgb_cap.read()
        ret_thermal, thermal_frame = thermal_cap.read()

        if not ret_rgb or not ret_thermal:
            break

        # Resize and normalize
        rgb_frame = cv2.resize(rgb_frame, (640, 480)).astype(np.float32) / 255.0
        thermal_frame = cv2.resize(thermal_frame, (640, 480)).astype(np.float32) / 255.0

        # Convert to the appropriate format for the model
        rgb_frame = np.transpose(rgb_frame, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        thermal_frame = np.expand_dims(thermal_frame, axis=0)  # (H, W) -> (1, H, W)

        # Convert to tensors
        rgb_frames.append(torch.tensor(rgb_frame))
        thermal_frames.append(torch.tensor(thermal_frame))

    return rgb_frames, thermal_frames, num_frames


def inference(rgb_frames, thermal_frames, num_frames, model, device):
    # Stack frames to create tensors of shape (num_frames, C, H, W)
    rgb_frames = torch.stack(rgb_frames).unsqueeze(0)  # Add batch dimension
    thermal_frames = torch.stack(thermal_frames).unsqueeze(0)  # Add batch dimension

    # Move data to the appropriate device
    rgb_frames = rgb_frames.to(device)
    thermal_frames = thermal_frames.to(device)

    # Forward pass through the model
    model.eval()
    with torch.no_grad():
        outputs = model(rgb_frames, thermal_frames, lengths=[num_frames])

    return outputs


if __name__ == '__main__':
    rgb_video_path = '/path/to/rgb/video.mp4'
    thermal_video_path = '/path/to/thermal/video.mp4'
    num_classes = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model
    rtf_net = RTFNet(n_class=num_classes).to(device)

    # Process the video and get predictions
    rgb_frames, thermal_frames, num_frames = process_video(rgb_video_path, thermal_video_path)
    outputs = inference(rgb_frames, thermal_frames, num_frames, rtf_net, device)

    # Post-process the output (e.g., take the last frame's prediction)
    _, predicted = torch.max(outputs, 1)
    print(f'Predicted class: {predicted.item()}')