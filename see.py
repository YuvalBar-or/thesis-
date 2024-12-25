import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Path to the folder and ending number
folder_path = 'CBIS-DDSM/Calc-Training_P_00005_RIGHT_MLO_1'
ending_number = '1'

def load_dicom_image(dicom_path):
    try:
        dicom_data = pydicom.dcmread(dicom_path)
        image = dicom_data.pixel_array
        return image
    except Exception as e:
        print(f"Error loading DICOM file {dicom_path}: {e}")
        return None

def display_images(grayscale_image, mask):
    plt.figure(figsize=(12, 6))

    # Display the grayscale image
    plt.subplot(1, 2, 1)
    plt.title("Grayscale Image")
    plt.imshow(grayscale_image, cmap='gray')
    plt.axis('off')

    # Display the mask
    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.show()

def main():
    # Define paths for grayscale image and mask
    grayscale_image_path = None
    mask_paths = []

    # Search for DICOM files
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dcm"):
                full_path = os.path.join(root, file)
                if f"_{ending_number}" in root:
                    mask_paths.append(full_path)
                else:
                    grayscale_image_path = full_path

    if not grayscale_image_path:
        print("No grayscale image found.")
        return

    if not mask_paths:
        print("No mask files found.")
        return

    # Load the grayscale image
    grayscale_image = load_dicom_image(grayscale_image_path)
    if grayscale_image is None:
        print("Failed to load the grayscale image.")
        return

    # Select the mask with the most black pixels
    selected_mask = None
    max_black_pixels = -1
    for mask_path in mask_paths:
        mask = load_dicom_image(mask_path)
        if mask is not None:
            black_pixels = np.sum(mask == 0)
            if black_pixels > max_black_pixels:
                max_black_pixels = black_pixels
                selected_mask = mask

    if selected_mask is None:
        print("Failed to load a valid mask.")
        return

    # Normalize and binarize the mask
    selected_mask = (selected_mask > 0).astype(np.uint8)

    # Display the images
    display_images(grayscale_image, selected_mask)

if __name__ == "__main__":
    main()
