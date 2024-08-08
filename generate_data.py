import cv2
import numpy as np


def display_dataset():
    img_path = "/Users/mikky/Downloads/dataset/images/01353N.png"
    label_path = "/Users/mikky/Downloads/dataset/labels/01353N.png"

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    print(img.shape)
    print (img)
    print (label)


def sample_frames(video_path, num_frames=3):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Failed to read frame at index {idx}")

    cap.release()
    return frames


def combine_frames(rgb_frame, thermal_frame):
    combined_frame = np.dstack((rgb_frame, thermal_frame))
    return combined_frame


def save_combined_frames(combined_frames, output_dir, CALIBRATION):
    for i, frame in enumerate(combined_frames):
        calibration = '_calibration' if CALIBRATION else ''
        output_path = f'{output_dir}/combined_frame_{i + 1}{calibration}.png'
        cv2.imwrite(output_path, frame)
        print(f'Saved {output_path}')


def preprocess_thermal(rgb_img, thermal_img, calibration=True):
    """
    :param rgb_img: rgb_img is for calibration
    :param thermal_img:
    :return:
    """
    # Grayscale thermal
    thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2GRAY)
    # Calibration
    if calibration:
        ribboned_thermal_img = calibrate_thermal(rgb_img, thermal_img)
    else:
        ribboned_thermal_img = cv2.resize(thermal_img, (640, 480), interpolation=cv2.INTER_LINEAR)
    return ribboned_thermal_img


def calibrate_thermal(rgb_img, thermal_img):
    height, width, _ = rgb_img.shape

    # Define surrounding ribbon of pixels from around thermal image (thermal camera's blind points)
    ribbon_offset_x_left = 65
    ribbon_offset_x_right = 100
    ribbon_offset_y_top = 65
    ribbon_offset_y_bottom = 65

    # Create a black image with the same size as the RGB image
    black_image = np.zeros((height, width), dtype=np.uint8)

    # Upscale the thermal (32x24) to the size of the area within the RGB image minus the ribbon
    thermal_img_resized = cv2.resize(thermal_img.reshape(24, 32),
                                     (width - (ribbon_offset_x_left + ribbon_offset_x_right),
                                      height - (ribbon_offset_y_top + ribbon_offset_y_bottom)),
                                     interpolation=cv2.INTER_LINEAR)

    # Place the resized thermal image into the black image at the correct position
    black_image[ribbon_offset_y_top:height - ribbon_offset_y_bottom,
    ribbon_offset_x_left:width - ribbon_offset_x_right] = thermal_img_resized

    return black_image


def generate_test_images():
    # Paths to your RGB and thermal videos
    rgb_video_path = '/Users/mikky/Downloads/1677760130200_1677760134400_18_rgb.mp4'
    thermal_video_path = '/Users/mikky/Downloads/1677760130200_1677760134400_18_thermal.mp4'

    # Sample frames
    rgb_frames = sample_frames(rgb_video_path)
    thermal_frames = sample_frames(thermal_video_path)

    CALIBRATION = False
    # Preprocess thermal images
    thermal_frames = [preprocess_thermal(rgb, thermal, calibration=CALIBRATION) for rgb, thermal in zip(rgb_frames, thermal_frames)]

    # Step 2: Combine Frames
    combined_frames = [combine_frames(rgb, thermal) for rgb, thermal in zip(rgb_frames, thermal_frames)]

    # Step 3: Save as PNG
    output_dir = 'dataset/images'
    save_combined_frames(combined_frames, output_dir, CALIBRATION)

generate_test_images()



