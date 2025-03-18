import SimpleITK as sitk
import numpy as np

# Load DICOM series
dicom_folder = r"D:\opera\ORIG_3D_FSPGR_20_Average\ORIG_3D_FSPGR_20_Average"
reader = sitk.ImageSeriesReader()
reader.SetFileNames(reader.GetGDCMSeriesFileNames(dicom_folder))
raw_img_sitk = reader.Execute()

# Convert image to float32
raw_img_sitk = sitk.Cast(raw_img_sitk, sitk.sitkFloat32)

# Apply Otsu thresholding first
otsu_mask = sitk.OtsuThreshold(raw_img_sitk, 0, 1)
otsu_mask_array = sitk.GetArrayFromImage(otsu_mask)

# Apply N4 bias field correction
bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
bias_corrected = bias_corrector.Execute(raw_img_sitk)

# Apply thresholding on the bias-corrected image
threshold_mask = sitk.OtsuThreshold(bias_corrected, 0, 1)
threshold_mask_array = sitk.GetArrayFromImage(threshold_mask)

# Convert bias-corrected image to numpy array
bias_corrected_array = sitk.GetArrayFromImage(bias_corrected)

# Stack bias-corrected image and threshold mask into a single 4D array
combined_array = np.stack((bias_corrected_array, threshold_mask_array), axis=0)

# Convert back to a SimpleITK image
combined_image = sitk.GetImageFromArray(combined_array)

# Set metadata from the original image (spacing, origin, direction)
combined_image.SetSpacing(raw_img_sitk.GetSpacing())
combined_image.SetOrigin(raw_img_sitk.GetOrigin())
combined_image.SetDirection(raw_img_sitk.GetDirection())

# Save combined image as NIfTI
output_path = r"D:\opera\ORIG_3D_FSPGR_20_Average\output_image.nii.gz"
sitk.WriteImage(combined_image, output_path)

print(f"Saved combined NIfTI file: {output_path}")
