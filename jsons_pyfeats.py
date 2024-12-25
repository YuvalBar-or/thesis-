import os
import json
import re
import numpy as np
import pandas as pd
import pydicom
import cv2
import pyfeats
import matplotlib.pyplot as plt

# Paths
meta_path = 'CBIS-DDSM'
csv_files = {
    'calc_train': 'csv/calc_case_description_train_set.csv',
    'calc_test': 'csv/calc_case_description_test_set.csv',
    'mass_train': 'csv/mass_case_description_train_set.csv',
    'mass_test': 'csv/mass_case_description_test_set.csv'
}
output_json_path = 'Extracted_Feature_pyfeats_JSON'

# Ensure the output directory exists
os.makedirs(output_json_path, exist_ok=True)

# Suppress warnings globally
np.seterr(divide='ignore', invalid='ignore')


# Function to normalize the mask
def normalize_mask(mask):
    return mask / np.max(mask) if np.max(mask) != 0 else mask


# Function to calculate the perimeter of the mask
def calculate_perimeter(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(perimeter_mask, contours, -1, 1, thickness=1)
    return perimeter_mask


import gc
import cv2
import numpy as np

import matplotlib.pyplot as plt

def extract_features_with_pyfeats(f, mask, perimeter):
    try:
        features = {}
        # Check if the mask is empty or invalid
        if np.count_nonzero(mask) == 0:
            print("Warning: Mask is empty or invalid. Skipping feature extraction.")
            return {}

        # Downscale images to reduce memory usage (e.g., resize to 256x256)
        f = cv2.resize(f, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        # Normalize the grayscale image
        f = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Normalize the mask
        mask = mask / np.max(mask) if np.max(mask) != 0 else mask

        print("\nStarting feature extraction with PyFeats...")
        print(f"Grayscale image shape: {f.shape}, Mask shape: {mask.shape}")

        # Extract features
        feature_extraction_functions = [
            ("FOS", pyfeats.fos, [f, mask]),
            ("GLCM", pyfeats.glcm_features, [f, True]),
            ("GLDS", pyfeats.glds_features, [f, mask, [0, 1, 1, 1], [1, 1, 0, -1]]),
            ("NGTDM", pyfeats.ngtdm_features, [f, mask, 1]),
            ("SFM", pyfeats.sfm_features, [f, mask, 4, 4]),
            ("LTE", pyfeats.lte_measures, [f, mask, 7]),
            ("FDTA", pyfeats.fdta, [f, mask, 3]),
            ("GLRLM", pyfeats.glrlm_features, [f, mask, 256]),
            ("FPS", pyfeats.fps, [f, mask]),
            ("Shape Parameters", pyfeats.shape_parameters, [f, mask, perimeter, 1]),
            ("GLSZM", pyfeats.glszm_features, [f, mask]),
            ("LBP", pyfeats.lbp_features, [f, mask, [8, 16, 24], [1, 2, 3]]),
            ("AMFM", pyfeats.amfm_features, [f, 32]),
            ("DWT", pyfeats.dwt_features, [f, mask, "bior3.3", 3]),
            ("SWT", pyfeats.swt_features, [f, mask, "bior3.3", 3]),
            ("GT", pyfeats.gt_features, [f, mask, 4, [0.05, 0.4]]),
            ("Zernike Moments", pyfeats.zernikes_moments, [f, 9]),
            ("Hu Moments", pyfeats.hu_moments, [f]),
            ("TAS", pyfeats.tas_features, [f]),
        ]

        for name, func, args in feature_extraction_functions:
            try:
                result = func(*args)
                if isinstance(result, tuple) and len(result) == 2:
                    labels, values = result
                    features.update(dict(zip(labels, values)))
                print(f"{name} features extracted.")
            except Exception as e:
                print(f"Error in {name}: {e}")

        print("\nFeature extraction completed successfully.")
        return features
    except Exception as e:
        print(f"General error extracting features with PyFeats: {e}")
        return {}



def extract_limited_features_with_pyfeats(mask, perimeter):
    try:
        features = {}
        print("\nGrayscale image not found. Extracting limited features...")

        # Normalize the mask
        mask = normalize_mask(mask)

        # Extract Zernikes' Moments
        try:
            zernike_features, zernike_labels = pyfeats.zernikes_moments(mask, radius=9)
            features.update(dict(zip(zernike_labels, zernike_features)))
            print(f"Zernikes' Moments extracted: {len(zernike_features)}")
        except Exception as e:
            print(f"Error in Zernikes' Moments: {e}")

        # Extract Hu's Moments
        try:
            hu_features, hu_labels = pyfeats.hu_moments(mask)
            features.update(dict(zip(hu_labels, hu_features)))
            print(f"Hu's Moments extracted: {len(hu_features)}")
        except Exception as e:
            print(f"Error in Hu's Moments: {e}")

        # Extract Shape Parameters
        try:
            shape_features, shape_labels = pyfeats.shape_parameters(None, mask, perimeter, pixels_per_mm2=1)
            features.update(dict(zip(shape_labels, shape_features)))
            print(f"Shape Parameters extracted: {len(shape_features)}")
        except Exception as e:
            print(f"Error in Shape Parameters: {e}")

        return features
    except Exception as e:
        print(f"General error in limited feature extraction: {e}")
        return {}


def get_original_image_path(folder_name_base):
    """
    Retrieve the path to the original grayscale DICOM image based on the folder_name_base.
    Ensures that the image is a valid grayscale image (contains multiple gray levels).
    """
    for root, _, files in os.walk(meta_path):
        if folder_name_base in root:
            for file in files:
                if file.endswith(".dcm"):
                    dicom_path = os.path.join(root, file)
                    try:
                        dicom_data = pydicom.dcmread(dicom_path)
                        pixel_array = dicom_data.pixel_array
                        unique_values = np.unique(pixel_array)

                        # Check if the image has multiple gray levels
                        if len(unique_values) > 2:
                            return dicom_path
                    except Exception as e:
                        print(f"Error reading DICOM file {dicom_path}: {e}")
    return None


# Function to extract DICOM paths for folders ending with a number
def get_all_dicom_paths(row, is_train, abnormality_type):
    folder_type = 'Training' if is_train else 'Test'
    folder_name_base = f"{abnormality_type.capitalize()}-{folder_type}_{row['patient_id']}_{row['left or right breast']}_{row['image view']}"
    pattern = re.compile(f"{re.escape(folder_name_base)}_([0-9]+)$")  # Capture the ending number
    matching_dirs = []

    for root, dirs, _ in os.walk(meta_path):
        for d in dirs:
            match = pattern.match(d)
            if match:
                full_path = os.path.join(root, d)
                folder_number = match.group(1)
                print(f"Processing folder: {full_path} with ending number: {folder_number}")
                matching_dirs.append((full_path, folder_number))  # Include the ending number

    if not matching_dirs:
        print(f"No matching directories found for base folder: {folder_name_base}")
        return []
    return matching_dirs


def create_birads_json_files(df, is_train, abnormality_type):
    for index, row in df.iterrows():
        try:
            # Extract relevant fields
            patient_id = row['patient_id']

            # Get the original image path
            folder_type = 'Training' if is_train else 'Test'
            folder_name_base = f"{abnormality_type.capitalize()}-{folder_type}_{row['patient_id']}_{row['left or right breast']}_{row['image view']}"
            original_image_path = get_original_image_path(folder_name_base)

            if not original_image_path:
                print(f"Warning: No grayscale image found for patient {patient_id} in folder {folder_name_base}. Using limited feature extraction.")
                matching_dirs = get_all_dicom_paths(row, is_train, abnormality_type)
                if not matching_dirs:
                    print(f"No DICOM mask files found for patient {patient_id}")
                    continue

                for dicom_folder_path, folder_number in matching_dirs:
                    # Gather all DICOM files in this folder
                    dicom_files = []
                    for root, _, files in os.walk(dicom_folder_path):
                        for file in files:
                            if file.endswith(".dcm"):
                                dicom_files.append(os.path.join(root, file))

                    # Select the mask file with the most black pixels
                    selected_mask_file = select_image_with_more_black_pixels(dicom_files)
                    if selected_mask_file is None:
                        print(f"No valid mask found for patient {patient_id} in folder {dicom_folder_path}")
                        continue

                    # Load the mask
                    mask_data = pydicom.dcmread(selected_mask_file)
                    mask = (mask_data.pixel_array > 0).astype(np.uint8)  # Binarize the mask
                    perimeter = calculate_perimeter(mask)

                    # Extract limited features
                    mask_features = extract_limited_features_with_pyfeats(mask, perimeter)

                    # Construct JSON file name and path with the ending number
                    json_filename = f"{folder_name_base}_{folder_number}.json".replace(" ", "_")
                    json_filepath = os.path.join(output_json_path, json_filename)

                    # Prepare data for JSON by excluding 'pathology' and specific path fields
                    json_data = row.drop(
                        labels=['pathology', 'assessment', 'image file path', 'cropped image file path',
                                'ROI mask file path']
                    ).to_dict()

                    # Add limited features
                    json_data.update(mask_features)

                    # Save data to JSON
                    with open(json_filepath, 'w') as json_file:
                        json.dump(json_data, json_file, indent=4)

                    print(f"Created JSON file (limited features) for folder {dicom_folder_path} with number {folder_number} - {json_filepath}")

            else:
                print(f"Grayscale image found for patient {patient_id} in folder {folder_name_base}. Using full feature extraction.")
                matching_dirs = get_all_dicom_paths(row, is_train, abnormality_type)
                if not matching_dirs:
                    print(f"No DICOM mask files found for patient {patient_id}")
                    continue

                for dicom_folder_path, folder_number in matching_dirs:
                    # Gather all DICOM files in this folder
                    dicom_files = []
                    for root, _, files in os.walk(dicom_folder_path):
                        for file in files:
                            if file.endswith(".dcm"):
                                dicom_files.append(os.path.join(root, file))

                    # Select the mask file with the most black pixels
                    selected_mask_file = select_image_with_more_black_pixels(dicom_files)
                    if selected_mask_file is None:
                        print(f"No valid mask found for patient {patient_id} in folder {dicom_folder_path}")
                        continue

                    # Load the grayscale image and mask
                    original_image_data = pydicom.dcmread(original_image_path)
                    mask_data = pydicom.dcmread(selected_mask_file)

                    f = original_image_data.pixel_array  # Grayscale image
                    mask = (mask_data.pixel_array > 0).astype(np.uint8)  # Binarize the mask
                    perimeter = calculate_perimeter(mask)

                    # Extract full features
                    mask_features = extract_features_with_pyfeats(f, mask, perimeter)

                    # Construct JSON file name and path with the ending number
                    json_filename = f"{folder_name_base}_{folder_number}.json".replace(" ", "_")
                    json_filepath = os.path.join(output_json_path, json_filename)

                    # Prepare data for JSON by excluding 'pathology' and specific path fields
                    json_data = row.drop(
                        labels=['pathology', 'assessment', 'image file path', 'cropped image file path',
                                'ROI mask file path']
                    ).to_dict()

                    # Add full features
                    json_data.update(mask_features)

                    # Save data to JSON
                    with open(json_filepath, 'w') as json_file:
                        json.dump(json_data, json_file, indent=4)

                    print(f"Created JSON file (full features) for folder {dicom_folder_path} with number {folder_number} - {json_filepath}")

        except Exception as e:
            print(f"Failed to create JSON for Patient ID {patient_id} with error: {e}")



# Function to select the image with the most black pixels
def select_image_with_more_black_pixels(mask_paths, threshold=50):
    if len(mask_paths) < 2:
        print(f"Only one or no mask files found, selecting {mask_paths[0] if mask_paths else 'None'}")
        return mask_paths[0] if mask_paths else None
    try:
        black_ratios = []
        for mask_path in mask_paths:
            dicom_data = pydicom.dcmread(mask_path)
            pixel_array = dicom_data.pixel_array
            total_pixels = pixel_array.size
            black_pixels = np.sum(pixel_array < threshold)
            black_ratios.append(black_pixels / total_pixels)
        return mask_paths[np.argmax(black_ratios)]
    except Exception as e:
        print(f"Error selecting image with more black pixels: {e}")
        return None


# Load each CSV file and create JSON files accordingly
for key, csv_path in csv_files.items():
    print(f"\nProcessing {key.replace('_', ' ')} CSV file: {csv_path}")
    df = pd.read_csv(csv_path)

    # Determine if the file is for training or test and if it is mass or calc
    is_train = 'train' in key
    abnormality_type = 'calc' if 'calc' in key else 'mass'

    # Create JSON files based on the abnormality type
    create_birads_json_files(df, is_train=is_train, abnormality_type=abnormality_type)

print("\nAll JSON files created successfully in the Extracted_Feature_pyfeats_JSON folder.")
