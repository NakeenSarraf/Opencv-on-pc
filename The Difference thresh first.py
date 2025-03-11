import SimpleITK as sitk
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

# Rescale intensity to range 0-255 for better contrast (optional)
transformed = sitk.RescaleIntensity(raw_img_sitk, 0, 255)

# Apply Otsu’s thresholding to create a binary mask (segmentation)
mask = sitk.OtsuThreshold(transformed, 0, 1)

# Shrink the images for faster processing (optional)
shrinkFactor = 4
inputImage = sitk.Shrink(transformed, [shrinkFactor] * raw_img_sitk.GetDimension())
maskImage = sitk.Shrink(mask, [shrinkFactor] * raw_img_sitk.GetDimension())

# Perform N4 bias field correction
bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrected = bias_corrector.Execute(inputImage, maskImage)

# Resample the corrected image to match the size of the original image
resampler = sitk.ResampleImageFilter()
resampler.SetSize(raw_img_sitk.GetSize())  # Match the size of the original image
resampler.SetOutputSpacing(raw_img_sitk.GetSpacing())  # Match the spacing of the original image
resampler.SetOutputOrigin(raw_img_sitk.GetOrigin())  # Match the origin of the original image
resampler.SetSize(raw_img_sitk.GetSize())  # Set the output size to match the original
corrected_resampled = resampler.Execute(corrected)

# Convert images to NumPy arrays for visualization
original_array = sitk.GetArrayViewFromImage(raw_img_sitk)
corrected_array = sitk.GetArrayViewFromImage(corrected_resampled)

# Compute the difference
difference = original_array - corrected_array

# Select the 22nd slice (index 21) for visualization
slice_idx = 21  # Fixed index for the 22nd slice

# Plot side by side
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(original_array[slice_idx], cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(corrected_array[slice_idx], cmap='gray')
plt.title("Bias Field Corrected Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(difference[slice_idx], cmap='coolwarm')
plt.title("Difference (Original - Corrected)")
plt.axis("off")

plt.show()
