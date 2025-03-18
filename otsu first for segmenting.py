import SimpleITK as sitk
import numpy as np
import os

# Load original DICOM series
dicom_folder = r"D:\opera\ORIG_3D_FSPGR_20_Average\ORIG_3D_FSPGR_20_Average"
reader = sitk.ImageSeriesReader()
reader.SetFileNames(reader.GetGDCMSeriesFileNames(dicom_folder))
raw_img_sitk = reader.Execute()

# Convert to float32
raw_img_sitk = sitk.Cast(raw_img_sitk, sitk.sitkFloat32)
transformed = sitk.RescaleIntensity(raw_img_sitk, 0, 255)

# Apply Otsu thresholding first
otsu_mask = sitk.OtsuThreshold(transformed, 0, 1)

# Perform N4 bias field correction
bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrected = bias_corrector.Execute(transformed, otsu_mask)

# Save as NIfTI (.nii)
output_dir = r"D:\opera\ORIG_3D_FSPGR_20_Average"
os.makedirs(output_dir, exist_ok=True)

# Save the corrected image
corrected_nii_path = os.path.join(output_dir, "bias_corrected.nii")
sitk.WriteImage(corrected, corrected_nii_path)
print(f"Saved bias-corrected image: {corrected_nii_path}")

# Save the Otsu mask
otsu_mask_nii_path = os.path.join(output_dir, "otsu_mask.nii")
sitk.WriteImage(otsu_mask, otsu_mask_nii_path)
print(f"Saved Otsu mask: {otsu_mask_nii_path}")