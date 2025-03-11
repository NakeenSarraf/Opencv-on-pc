import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

# Define the path to the DICOM folder
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

# Convert the image data type to Float32 for further processing
raw_img_sitk = sitk.Cast(raw_img_sitk, sitk.sitkFloat32)

# Rescale intensity to range 0-255 for better contrast
transformed = sitk.RescaleIntensity(raw_img_sitk, 0, 255)

# Apply Otsu’s thresholding to create a binary mask (segmentation)
mask = sitk.OtsuThreshold(transformed, 0, 1)

# Perform bias field correction using the N4BiasFieldCorrection filter
shrinkFactor = 4
inputImage = sitk.Shrink(transformed, [shrinkFactor] * raw_img_sitk.GetDimension())
maskImage = sitk.Shrink(mask, [shrinkFactor] * raw_img_sitk.GetDimension())

bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrected = bias_corrector.Execute(inputImage, maskImage)

# Get the log bias field and apply it to the full-resolution image
log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_img_sitk)
corrected_image_full_resolution = raw_img_sitk / sitk.Exp(log_bias_field)

# Convert images to NumPy arrays for visualization
original_array = sitk.GetArrayViewFromImage(raw_img_sitk)
corrected_array = sitk.GetArrayViewFromImage(corrected_image_full_resolution)

# Select the 22nd slice (index 21) for visualization
slice_idx = 21  # Fixed index for the 22nd slice

# Plot side by side
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(original_array[slice_idx], cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(corrected_array[slice_idx], cmap='gray')
plt.title("Bias Field Corrected Image")
plt.axis("off")

plt.show()