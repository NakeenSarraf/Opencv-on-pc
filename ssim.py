import SimpleITK as sitk
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Define the path to your DICOM folder
dicom_folder = r"D:\opera\ORIG_3D_FSPGR_20_Average\ORIG_3D_FSPGR_20_Average"

# Create a DICOM series reader
reader = sitk.ImageSeriesReader()

# Get the list of DICOM file names in the specified folder
dicom_series = reader.GetGDCMSeriesFileNames(dicom_folder)

# Set the DICOM file names for the reader
reader.SetFileNames(dicom_series)

# Load the DICOM series into a SimpleITK image object
raw_img_sitk = reader.Execute()

# Check if the image was loaded successfully
if raw_img_sitk.GetSize() == (0, 0, 0):
    raise ValueError("Failed to load DICOM series. Check the folder path.")

# Convert the image to a 32-bit float (ensure the input image is preprocessed correctly)
raw_img_sitk = sitk.Cast(raw_img_sitk, sitk.sitkFloat32)

# Perform N4 bias field correction
bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrected = bias_corrector.Execute(raw_img_sitk)

# Resample the corrected image to match the size of the original image
resampler = sitk.ResampleImageFilter()
resampler.SetSize(raw_img_sitk.GetSize())  # Match the size of the original image
resampler.SetOutputSpacing(raw_img_sitk.GetSpacing())  # Match the spacing of the original image
resampler.SetOutputOrigin(raw_img_sitk.GetOrigin())  # Match the origin of the original image
corrected_resampled = resampler.Execute(corrected)

# Rescale intensity of both images to the range [0, 255]
original_rescaled = sitk.RescaleIntensity(raw_img_sitk, 0, 255)
corrected_rescaled = sitk.RescaleIntensity(corrected_resampled, 0, 255)

# Convert images to NumPy arrays for SSIM calculation
original_array_rescaled = sitk.GetArrayViewFromImage(original_rescaled)
corrected_array_rescaled = sitk.GetArrayViewFromImage(corrected_rescaled)

# Compute SSIM with rescaled images
ssim_index = ssim(original_array_rescaled, corrected_array_rescaled, multichannel=False, data_range=255)

# Print SSIM value
print(f"SSIM between original and bias-corrected image: {ssim_index}")