import SimpleITK as sitk
import matplotlib.pyplot as plt

# Define the path to your DICOM folder
dicom_folder = r"D:\opera\ORIG_3D_FSPGR_20_Average\ORIG_3D_FSPGR_20_Average"

# Read DICOM series
reader = sitk.ImageSeriesReader()
dicom_series = reader.GetGDCMSeriesFileNames(dicom_folder)
reader.SetFileNames(dicom_series)
raw_img_sitk = reader.Execute()

if raw_img_sitk.GetSize() == (0, 0, 0):
    raise ValueError("Failed to load DICOM series. Check the folder path.")

# Convert to float32 for processing
raw_img_sitk = sitk.Cast(raw_img_sitk, sitk.sitkFloat32)

# Perform N4 bias field correction FIRST
bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrected = bias_corrector.Execute(raw_img_sitk)

# Apply Otsu’s thresholding AFTER bias correction
thresholded = sitk.OtsuThreshold(corrected, 0, 1)

# Convert images to NumPy arrays
original_array = sitk.GetArrayViewFromImage(raw_img_sitk)
corrected_array = sitk.GetArrayViewFromImage(corrected)
thresholded_array = sitk.GetArrayViewFromImage(thresholded)

# Compute difference
difference = original_array - corrected_array

# Select slice 22 (index 21), ensuring it exists
slice_idx = min(21, original_array.shape[0] - 1)  # Ensures within range

# Plot side by side
plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.imshow(original_array[slice_idx], cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(corrected_array[slice_idx], cmap='gray')
plt.title("Bias Field Corrected")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(thresholded_array[slice_idx], cmap='gray')
plt.title("Thresholded (After Bias Correction)")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(difference[slice_idx], cmap='coolwarm')
plt.title("Difference (Original - Corrected)")
plt.axis("off")

plt.show()
