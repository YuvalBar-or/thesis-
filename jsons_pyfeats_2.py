import os
import json
import re
import numpy as np
import pandas as pd
import pydicom
import cv2
import pyfeats
from multiprocessing import Pool
from functools import lru_cache

# Paths
meta_path = 'CBIS-DDSM'
csv_files = {
    'calc_train': 'csv/calc_case_description_train_set.csv',
    'calc_test': 'csv/calc_case_description_test_set.csv',
    'mass_train': 'csv/mass_case_description_train_set.csv',
    'mass_test': 'csv/mass_case_description_test_set.csv'
}
output_json_path = 'Extracted_Feature_pyfeats_JSON'
os.makedirs(output_json_path, exist_ok=True)

# Suppress warnings globally
np.seterr(divide='ignore', invalid='ignore')


# Helper functions
def normalize_mask(mask):
    return mask / np.max(mask) if np.max(mask) != 0 else mask


def calculate_perimeter(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(perimeter_mask, contours, -1, 1, thickness=1)
    return perimeter_mask


@lru_cache(maxsize=None)
def get_dicom_data(dicom_path):
    dicom_data = pydicom.dcmread(dicom_path)
    return dicom_data.pixel_array


@lru_cache(maxsize=None)
def get_original_image_path(folder_name_base):
    for root, _, files in os.walk(meta_path):
        if folder_name_base in root:
            for file in files:
                if file.endswith(".dcm"):
                    dicom_path = os.path.join(root, file)
                    try:
                        dicom_data = pydicom.dcmread(dicom_path)
                        pixel_array = dicom_data.pixel_array
                        unique_values = np.unique(pixel_array)
                        if len(unique_values) > 2:  # Check for multiple gray levels
                            return dicom_path
                    except Exception as e:
                        print(f"Error reading DICOM file {dicom_path}: {e}")
    return None


def select_image_with_more_black_pixels(mask_paths, threshold=50):
    try:
        black_ratios = [
            (path, np.sum(get_dicom_data(path) < threshold) / get_dicom_data(path).size)
            for path in mask_paths
        ]
        return max(black_ratios, key=lambda x: x[1])[0]
    except Exception as e:
        print(f"Error selecting image with more black pixels: {e}")
        return None


def extract_features(image, mask, perimeter):
    features = {}
    try:
        # Full feature extraction list
        feature_extraction_functions = [
            ("FOS", pyfeats.fos, [image, mask]),
            ("GLCM", pyfeats.glcm_features, [image, True]),
            ("GLDS", pyfeats.glds_features, [image, mask, [0, 1, 1, 1], [1, 1, 0, -1]]),
            ("NGTDM", pyfeats.ngtdm_features, [image, mask, 1]),
            ("SFM", pyfeats.sfm_features, [image, mask, 4, 4]),
            ("LTE", pyfeats.lte_measures, [image, mask, 7]),
            ("FDTA", pyfeats.fdta, [image, mask, 3]),
            ("GLRLM", pyfeats.glrlm_features, [image, mask, 256]),
            ("FPS", pyfeats.fps, [image, mask]),
            ("Shape Parameters", pyfeats.shape_parameters, [image, mask, perimeter, 1]),
            ("GLSZM", pyfeats.glszm_features, [image, mask]),
            ("LBP", pyfeats.lbp_features, [image, mask, [8, 16, 24], [1, 2, 3]]),
            ("AMFM", pyfeats.amfm_features, [image, 32]),
            ("DWT", pyfeats.dwt_features, [image, mask, "bior3.3", 3]),
            ("SWT", pyfeats.swt_features, [image, mask, "bior3.3", 3]),
            ("GT", pyfeats.gt_features, [image, mask, 4, [0.05, 0.4]]),
            ("Zernike Moments", pyfeats.zernikes_moments, [image, 9]),
            ("Hu Moments", pyfeats.hu_moments, [image]),
            ("TAS", pyfeats.tas_features, [image]),
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

    except Exception as e:
        print(f"Error extracting features: {e}")

    return features


def process_patient_row(row, is_train, abnormality_type):
    patient_id = row['patient_id']
    folder_type = 'Training' if is_train else 'Test'
    folder_name_base = f"{abnormality_type.capitalize()}-{folder_type}_{patient_id}_{row['left or right breast']}_{row['image view']}"
    original_image_path = get_original_image_path(folder_name_base)
    matching_dirs = get_matching_dirs(folder_name_base)

    for dicom_folder_path, folder_number in matching_dirs:
        dicom_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(dicom_folder_path)
            for file in files if file.endswith(".dcm")
        ]
        selected_mask_file = select_image_with_more_black_pixels(dicom_files)
        if not selected_mask_file:
            continue

        mask = (get_dicom_data(selected_mask_file) > 0).astype(np.uint8)
        perimeter = calculate_perimeter(mask)

        if original_image_path:
            image = get_dicom_data(original_image_path)
            image = cv2.normalize(cv2.resize(image, (256, 256)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            mask = cv2.resize(mask, (256, 256))
            mask = normalize_mask(mask)
            features = extract_features(image, mask, perimeter)
        else:
            features = {"error": "Grayscale image not found"}

        json_filename = f"{folder_name_base}_{folder_number}.json".replace(" ", "_")
        json_filepath = os.path.join(output_json_path, json_filename)

        json_data = row.drop(
            labels=['pathology', 'assessment', 'image file path', 'cropped image file path', 'ROI mask file path']
        ).to_dict()
        json_data.update(features)

        with open(json_filepath, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
        print(f"Created JSON: {json_filepath}")


def get_matching_dirs(folder_name_base):
    pattern = re.compile(f"{re.escape(folder_name_base)}_([0-9]+)$")
    matching_dirs = []
    for root, dirs, _ in os.walk(meta_path):
        for d in dirs:
            if pattern.match(d):
                matching_dirs.append((os.path.join(root, d), d.split('_')[-1]))
    return matching_dirs


def create_json_files(df, is_train, abnormality_type):
    rows = [(row, is_train, abnormality_type) for _, row in df.iterrows()]
    with Pool(processes=os.cpu_count()) as pool:
        pool.starmap(process_patient_row, rows)


if __name__ == "__main__":
    # Main processing loop
    for key, csv_path in csv_files.items():
        print(f"\nProcessing {key.replace('_', ' ')}: {csv_path}")
        df = pd.read_csv(csv_path)
        is_train = 'train' in key
        abnormality_type = 'calc' if 'calc' in key else 'mass'
        create_json_files(df, is_train, abnormality_type)

    print("\nJSON creation completed.")
