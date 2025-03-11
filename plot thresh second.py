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

# Perform N4 bias field correction (correcting the raw image)
bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrected = bias_corrector.Execute(raw_img_sitk)

# Get the log bias field after the correction
log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_img_sitk)

# Rescale intensity to range 0-255 for better contrast after bias correction (optional)
corrected_rescaled = sitk.RescaleIntensity(corrected, 0, 255)

# Apply Otsu’s thresholding to create a binary mask (segmentation) after correction
thresholded = sitk.OtsuThreshold(corrected_rescaled, 0, 1)

# Convert images to NumPy arrays for visualization
original_array = sitk.GetArrayFromImage(raw_img_sitk)
corrected_array = sitk.GetArrayFromImage(corrected)
thresholded_array = sitk.GetArrayFromImage(thresholded)

# Ensure the corrected image is resized to match the original image size
slice_idx = 21  # Fixed index for the 22nd slice

# Create a plot to visualize the results
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

# Thresholded image (segmentation result)
plt.subplot(1, 3, 3)
plt.imshow(thresholded_array[slice_idx], cmap='gray')
plt.title("Thresholded Image")
plt.axis("off")

# Show the color bar for bias field visualization
plt.figure(figsize=(6, 6))
plt.imshow(sitk.GetArrayFromImage(log_bias_field)[slice_idx], cmap='jet')  # Display using 'jet' colormap
plt.title("Estimated Bias Field")
plt.colorbar()  # Show the color scale for intensity values
plt.axis("off")  # Hide the axis for better visualization

# Show all the plots
plt.show()
