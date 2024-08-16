import os
import cv2
import numpy as np
import pandas as pd
import multiprocessing


def mean_valids(thermal_data):
    """
    Remove outliers
    """
    original = thermal_data.copy()
    findMean = thermal_data.copy()

    num_invalid = 0
    num_invalid = num_invalid + len(findMean[findMean < -40]) + len(findMean[findMean > 300])

    if num_invalid == 768:
        return None
    elif num_invalid != 0:
        findMean[findMean < -40] = 0
        findMean[findMean > 300] = 0
        adjusted_mean = np.sum(findMean) / (768 - num_invalid)

        original[original < -40] = adjusted_mean
        original[original > 300] = adjusted_mean

    return original


def preprocess_thermal(rgb_img, thermal_img, calibration=False):
    """
    :param rgb_img: rgb_img is for calibration
    :param thermal_img:
    :return:
    """
    # 1. Remove outliers
    thermal_img = mean_valids(thermal_img)
    if thermal_img is None:
        return None
    # 2. Normalize the value within 0 - 255
    thermal_img = cv2.normalize(thermal_img, thermal_img, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # 3. Make the person face up
    thermal_img = cv2.rotate(thermal_img, cv2.ROTATE_180)
    # 4. Calibrate the thermal image to match rgb image
    if calibration:
        ribboned_thermal_img = calibrate_thermal(rgb_img, thermal_img)
    else:
        ribboned_thermal_img = thermal_img
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
                                     interpolation=cv2.INTER_NEAREST)

    # Place the resized thermal image into the black image at the correct position
    black_image[ribbon_offset_y_top:height - ribbon_offset_y_bottom,
    ribbon_offset_x_left:width - ribbon_offset_x_right] = thermal_img_resized

    return black_image


def process_data(pid, input_csv, output_dir, data_split):
    # Read video clips info
    clips_csv = os.path.join(input_csv.format(pid=pid), data_split + ".csv")
    clips_df = pd.read_csv(clips_csv, names=['clip_path', 'label'])

    # Define label dataframe
    label_list = []

    # Loop through clips_df
    for idx, row in clips_df.iterrows():
        # Get start and end timestamps
        clip_path = row['clip_path']
        data_pid = int(clip_path.split('/')[3].split('P')[1])
        label = row['label']
        start = int(clip_path.split('/')[-1].split('_')[0])
        end = int(clip_path.split('/')[-1].split('_')[1])
        # if start == 1681811689600 and end == 1681811692200 and data_pid == 14:

        ts_df = pd.read_csv(os.path.join(DIR_MAP[data_pid], 'timestamps.csv'))
        # Get start and end frames
        filtered_df = ts_df[(ts_df['timestamp'] >= start) & (ts_df['timestamp'] <= end)]
        rgb_frames = filtered_df['rgb_path'].sort_values().tolist()
        thermal_npy_frames = filtered_df['thermal_numpy_path'].sort_values().tolist()

        if len(rgb_frames) == 0:
            print(f"***** No frames between start and end! Error here! ***** start: ",  start, 'end: ', end)
            continue
        if len(rgb_frames) != len(thermal_npy_frames):
            print(f"***** RGB and Thermal sequence length do not match. *****")
            continue

        # Create RGB-T frames
        rgb_array = []
        thermal_array = []
        rgb_filenames = []  # Store original RGB filenames
        thermal_filenames = []  # Store original thermal filenames

        for idx, file_name in enumerate(thermal_npy_frames):
            rgb_image_path = rgb_frames[idx]
            thermal_npy_path = thermal_npy_frames[idx]
            try:
                bgr_image = cv2.imread(os.path.join(DIR_MAP[data_pid], rgb_image_path))
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                rgb_image = cv2.rotate(rgb_image, cv2.ROTATE_180)
                height, width, _ = rgb_image.shape

                thermal_data = np.load(os.path.join(DIR_MAP[data_pid], thermal_npy_path))
                thermal_image = preprocess_thermal(rgb_image, thermal_data, calibration=False)
                thermal_image = cv2.resize(thermal_image, (width, height))
                if thermal_image is None:
                    # Skip when thermal image contains too many outliers
                    continue
            except Exception as e:
                print(e)
                continue
            rgb_array.append(rgb_image)
            thermal_array.append(thermal_image)
            rgb_filenames.append(os.path.basename(rgb_image_path))  # Store the original RGB filename
            # Change the thermal filename from .npy to .jpg
            thermal_filename = os.path.basename(thermal_npy_path)
            thermal_filename = os.path.splitext(thermal_filename)[0] + '.jpg'
            thermal_filenames.append(thermal_filename)  # Store the modified thermal filename

        image_output_dir = os.path.join(output_dir, 'P' + str(data_pid), 'rgbt-mid-fusion-rtfnet', 'image')
        sequence_name = f"{start}_{end}"
        rgb_output_dir = os.path.join(image_output_dir, sequence_name, 'rgb')
        thermal_output_dir = os.path.join(image_output_dir, sequence_name, 'thermal')
        # Create the directories if they don't exist
        os.makedirs(rgb_output_dir, exist_ok=True)
        os.makedirs(thermal_output_dir, exist_ok=True)

        # Save rgb and thermal images in respective folders
        for idx in range(len(rgb_array)):
            rgb_image = rgb_array[idx]
            thermal_image = thermal_array[idx]
            rgb_filename = rgb_filenames[idx]
            thermal_filename = thermal_filenames[idx]

            rgb_save_path = os.path.join(rgb_output_dir, rgb_filename)
            thermal_save_path = os.path.join(thermal_output_dir, thermal_filename)
            print(len(rgb_array), idx, rgb_save_path, thermal_save_path)
            # if thermal_save_path == '/ssd1/meixi/data/P14/rgbt-mid-fusion-rtfnet/image/1681811689600_1681811692200/thermal/1681811691400.jpg':
            # Save the RGB and thermal images
            cv2.imwrite(rgb_save_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving
            cv2.imwrite(thermal_save_path, thermal_image)

            # Logging
            print('participant id: ', pid, '; data split: ', data_split, '; data pid: ', data_pid,
                  '; label: ', label, '; index of videos: ', len(rgb_array))

        # Save the timestamp and label in the csv
        label_list.append((os.path.join(image_output_dir, sequence_name), label))

    # Save the timestamp and label in the csv
    label_df = pd.DataFrame(label_list)
    label_output_dir = os.path.join(output_dir, 'P' + str(pid), 'rgbt-mid-fusion-rtfnet', 'label')
    os.makedirs(label_output_dir, exist_ok=True)
    output_csv_path = os.path.join(label_output_dir, data_split + ".csv")
    label_df.to_csv(output_csv_path, header=False, index=False)


if __name__ == '__main__':
    DIR_MAP = {
        6: os.path.join("/ssd2", "behaviorsight", "data", "Wild", "P6", "output"),
        7: os.path.join("/ssd2", "behaviorsight", "data", "Wild", "P7", "output"),
        13: os.path.join("/ssd2", "behaviorsight", "data", "Wild", "P13", "output"),
        14: os.path.join("/ssd2", "behaviorsight", "data", "R21-InWild", "P14", "output"),
        15: os.path.join("/ssd2", "behaviorsight", "data", "R21-InWild", "P15", "output"),
        16: os.path.join("/ssd2", "behaviorsight", "data", "R21-InWild", "P16"),
        18: os.path.join("/ssd2", "behaviorsight", "data", "R21-InWild", "P18", "output"),
        1: os.path.join("/ssd2", "behaviorsight", "data", "Wild", "P1", "output"),
        2: os.path.join("/ssd2", "behaviorsight", "data", "R21-InWild", "P2", "output"),
        5: os.path.join("/ssd2", "behaviorsight", "data", "Wild", "P5", "output"),
        12: os.path.join("/ssd2", "behaviorsight", "data", "Wild", "P12", "output"),
        11: os.path.join("/ssd2", "behaviorsight", "data", "Wild", "P11", "output"),
        19: os.path.join("/ssd2", "behaviorsight", "data", "R21-InWild", "P19", "output"),
    }

    # # participant in batch
    smoking_pids = [6, 7, 13, 14, 15, 16, 18]
    data_splits = ['train', 'val', 'test']
    # output_dir = '/home/meixi/data'
    output_dir = '/ssd1/meixi/data'

    tasks = []
    for pid in smoking_pids:
        input_csv = f"/ssd2/R21_Clips/loso-data--for12participants/P{pid}/multi-rgb_losoBalanced"
        for data_split in data_splits:
            tasks.append((pid, input_csv, output_dir, data_split))

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(process_data, tasks)

    # tasks = []
    # for pid in smoking_pids:
    #     input_csv = f"/ssd2/R21_Clips/loso-data--for12participants/P{pid}/multi-rgb_losoBalanced"
    #     for data_split in data_splits:
    #         process_data(pid, input_csv, output_dir, data_split)
