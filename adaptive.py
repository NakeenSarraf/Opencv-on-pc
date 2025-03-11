import os
import numpy as np
import pydicom
import cv2 as cv
import nibabel as nib
from pydicom.pixel_data_handlers.util import apply_voi_lut
from glob import glob

# Path to the folder containing DICOM images
dicom_folder = "D:\opera\ORIG_3d_FSPGR_20_Average\ORIG_3d_FSPGR_20_Average"  # CHANGE THIS

# Read all DICOM files in the folder
dicom_files = sorted(glob(os.path.join(dicom_folder, "*.dcm")))

# Load DICOM slices
slices = [pydicom.dcmread(f) for f in dicom_files]
slices.sort(key=lambda x: x.InstanceNumber)  # Sort by instance number

# Convert DICOM to NumPy array (3D stack)
image_stack = np.stack([apply_voi_lut(s.pixel_array, s) for s in slices], axis=0)

# Normalize pixel values (scale between 0-255)
image_stack = (image_stack - np.min(image_stack)) / (np.max(image_stack) - np.min(image_stack)) * 255
image_stack = image_stack.astype(np.uint8)

# Apply adaptive thresholding to each slice
processed_stack = np.zeros_like(image_stack)

for i in range(image_stack.shape[0]):  # Iterate over slices
    img = cv.medianBlur(image_stack[i], 5)  # Denoise
    processed_stack[i] = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

# Convert to NIfTI
nii_image = nib.Nifti1Image(processed_stack, affine=np.eye(4))
nib.save(nii_image, "placenta_thresholded.nii")

print("Processing complete. Saved as 'placenta_thresholded.nii'")