import os
import numpy as np
import pydicom
import cv2 as cv
import nibabel as nib
from pydicom.pixel_data_handlers.util import apply_voi_lut
from glob import glob

dicom_folder = "D:\opera\ORIG_3d_FSPGR_20_Average\ORIG_3d_FSPGR_20_Average" 

dicom_files = sorted(glob(os.path.join(dicom_folder, "*.dcm")))

slices = [pydicom.dcmread(f) for f in dicom_files]
slices.sort(key=lambda x: x.InstanceNumber)

image_stack = np.stack([apply_voi_lut(s.pixel_array, s) for s in slices], axis=0)

image_stack = (image_stack - np.min(image_stack)) / (np.max(image_stack) - np.min(image_stack)) * 255
image_stack = image_stack.astype(np.uint8)

processed_stack = np.zeros_like(image_stack)

for i in range(image_stack.shape[0]):
    img = cv.GaussianBlur(image_stack[i], (5,5), 0) 
    _, processed_stack[i] = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

nii_image = nib.Nifti1Image(processed_stack, affine=np.eye(4))
nib.save(nii_image, "placenta_otsu.nii")

print("Processing complete. Saved as 'placenta_otsu.nii'")