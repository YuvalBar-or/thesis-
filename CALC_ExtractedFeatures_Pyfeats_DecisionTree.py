import os
import json
import re
import numpy as np
import pandas as pd
import pydicom
from pyfeats import shape_parameters

# Paths
meta_path = 'CBIS-DDSM'
csv_files = {
    'calc_train': 'csv/calc_case_description_train_set.csv',
    'calc_test': 'csv/calc_case_description_test_set.csv',
    'mass_train': 'csv/mass_case_description_train_set.csv',
    'mass_test': 'csv/mass_case_description_test_set.csv'
}
extracted_feature_json = 'Extracted_Feature_Pyfeats_JSON'

# Ensure the output directory exists
os.makedirs(extracted_feature_json, exist_ok=True)
print(f"Output directory created or already exists: {extracted_feature_json}")


# Function to calculate the percentage of black pixels in an image
def calculate_black_pixel_ratio(dicom_data, threshold=50):
    try:
        pixel_array = dicom_data.pixel_array
        total_pixels = pixel_array.size
        black_pixels = np.sum(pixel_array < threshold)
        return black_pixels / total_pixels
    except Exception as e:
        print(f"Error calculating black pixel ratio: {e}")
        return 0


# Function to select the image with the most black pixels
def select_image_with_more_black_pixels(mask_paths, threshold=50):
    if len(mask_paths) < 2:
        print(f"Only one or no mask files found, selecting {mask_paths[0] if mask_paths else 'None'}")
        return mask_paths[0] if mask_paths else None
    try:
        dicom_1 = pydicom.dcmread(mask_paths[0])
        dicom_2 = pydicom.dcmread(mask_paths[1])
        black_ratio_1 = calculate_black_pixel_ratio(dicom_1, threshold)
        black_ratio_2 = calculate_black_pixel_ratio(dicom_2, threshold)
        print(f"Black pixel ratio for {mask_paths[0]}: {black_ratio_1:.4f}")
        print(f"Black pixel ratio for {mask_paths[1]}: {black_ratio_2:.4f}")
        return mask_paths[0] if black_ratio_1 > black_ratio_2 else mask_paths[1]
    except Exception as e:
        print(f"Error selecting image with more black pixels: {e}")
        return None


# Updated `get_all_dicom_paths` function
def get_all_dicom_paths(row, is_train, abnormality_type):
    folder_type = 'Training' if is_train else 'Test'
    original_folder_base = f"{abnormality_type.capitalize()}-{folder_type}_{row['patient_id']}_{row['left or right breast']}_{row['image view']}"
    dicom_files = []
    mask_files = []

    # Find the original folder
    original_folder_path = None
    for root, dirs, _ in os.walk(meta_path):
        for dir_name in dirs:
            if dir_name == original_folder_base:
                original_folder_path = os.path.join(root, dir_name)

    if not original_folder_path:
        print(f"No original folder found for {original_folder_base}")
        return [], None

    # Find the corresponding mask folder
    mask_pattern = re.compile(f"{re.escape(original_folder_base)}_([0-9]+)$")
    for root, dirs, _ in os.walk(meta_path):
        for dir_name in dirs:
            if mask_pattern.match(dir_name):
                mask_files.append(os.path.join(root, dir_name))

    if not mask_files:
        print(f"No mask folders found for {original_folder_base}")
        return [], None

    # Select the mask with the most black pixels
    mask_paths = []
    for mask_folder in mask_files:
        for root, _, files in os.walk(mask_folder):
            for file in files:
                if file.endswith(".dcm"):
                    mask_paths.append(os.path.join(root, file))

    selected_mask = select_image_with_more_black_pixels(mask_paths) if mask_paths else None
    if not selected_mask:
        print(f"No valid mask found for {original_folder_base}")
        return [], None

    # Collect all DICOM files in the original folder
    for root, _, files in os.walk(original_folder_path):
        for file in files:
            if file.endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))

    return dicom_files, selected_mask


# Function to extract features using pyfeats
def extract_features(image, mask):
    try:
        # Ensure mask is binary
        mask = (mask > 0).astype(np.uint8)

        # Validate image and mask
        print(f"Original image shape: {image.shape}, Mask shape: {mask.shape}")
        print(f"Mask unique values: {np.unique(mask)}")
        print(f"Mask sum (number of pixels in ROI): {np.sum(mask > 0)}")

        # Ensure alignment
        if image.shape != mask.shape:
            print("Error: Image and mask shapes do not match.")
            return None

        # Check for empty masks
        if np.sum(mask) == 0:
            print("Mask is empty (no ROI detected). Features may be invalid.")
            return None

        # Extract features
        features, labels = shape_parameters(image, mask, perimeter=(mask > 0).astype(np.uint8))
        return dict(zip(labels, features))
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None



# Function to create a JSON file for each folder ending with a number
def create_extracted_feature_json_files(df, is_train, abnormality_type):
    print(f"Creating JSON files for {abnormality_type}...")
    for index, row in df.iterrows():
        try:
            patient_id = row['patient_id']
            print(f"Processing patient: {patient_id}")

            original_files, mask_path = get_all_dicom_paths(row, is_train, abnormality_type)
            if not original_files or not mask_path:
                print(f"No original or mask files found for patient {patient_id}")
                continue

            # Read the original image
            original_image_path = original_files[0]  # Assuming one file for simplicity
            original_dicom = pydicom.dcmread(original_image_path)
            original_image = original_dicom.pixel_array

            # Read the mask
            mask_dicom = pydicom.dcmread(mask_path)
            mask = mask_dicom.pixel_array

            # Extract features
            features = extract_features(original_image, mask)
            if features is None:
                print(f"Failed to extract features for patient {patient_id}")
                continue

            # Include patient ID in the JSON data
            features['patient_id'] = patient_id

            # Construct the JSON file name
            json_filename = f"{abnormality_type.capitalize()}-{row['left or right breast']}_{row['image view']}_{patient_id}.json"
            json_filepath = os.path.join(extracted_feature_json, json_filename)

            # Save the features as JSON
            with open(json_filepath, 'w') as json_file:
                json.dump(features, json_file, indent=4)

            print(f"Created JSON file: {json_filepath}")
        except Exception as e:
            print(f"Failed to create JSON for Patient ID {row['patient_id']}: {e}")


# Load each CSV file and create JSON files accordingly
for key, csv_path in csv_files.items():
    print(f"\nProcessing {key.replace('_', ' ')} CSV file: {csv_path}")
    df = pd.read_csv(csv_path)

    is_train = 'train' in key
    abnormality_type = 'calc' if 'calc' in key else 'mass'

    create_extracted_feature_json_files(df, is_train=is_train, abnormality_type=abnormality_type)

print("\nAll JSON files created successfully in the Extracted_Feature_Pyfeats_JSON folder.")
