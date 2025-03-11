import SimpleITK as sitk
import numpy as np
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

# Convert to 32-bit float for processing
raw_img_sitk = sitk.Cast(raw_img_sitk, sitk.sitkFloat32)

# Perform N4 bias field correction (correcting the raw image)
bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrected = bias_corrector.Execute(raw_img_sitk)

# Rescale intensity to 0-255 after bias correction
corrected_rescaled = sitk.RescaleIntensity(corrected, 0, 255)

# Apply Otsu’s thresholding
thresholded = sitk.OtsuThreshold(corrected_rescaled, 0, 1)

# Convert images to NumPy arrays for visualization
original_array = sitk.GetArrayFromImage(raw_img_sitk)
corrected_array = sitk.GetArrayFromImage(corrected)
thresholded_array = sitk.GetArrayFromImage(thresholded)

# Ensure the corrected image is resized to match the original image size
slice_idx = min(21, original_array.shape[0] - 1)  # Make sure slice index is within bounds

# Compute the difference between the original and corrected images
difference = original_array - corrected_array

# Plot side-by-side
plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(original_array[slice_idx], cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Bias field corrected image
plt.subplot(1, 3, 2)
plt.imshow(corrected_array[slice_idx], cmap='gray')
plt.title("Bias Field Corrected Image")
plt.axis("off")

# Difference (original - corrected)
plt.subplot(1, 3, 3)
plt.imshow(difference[slice_idx], cmap='coolwarm')
plt.title("Difference (Original - Corrected)")
plt.axis("off")

plt.show()
