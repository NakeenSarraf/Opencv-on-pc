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

# Get the log bias field
log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_img_sitk)

# Convert the log bias field to a NumPy array for plotting
bias_field_array = sitk.GetArrayViewFromImage(log_bias_field)

# Select the 22nd slice (index 21) for visualization
slice_idx = 21  # Fixed index for the 22nd slice

# Create a plot to visualize the bias field for the selected slice
plt.figure(figsize=(6, 6))
plt.imshow(bias_field_array[slice_idx], cmap='jet')  # Display using 'jet' colormap
plt.title("Estimated Bias Field")
plt.colorbar()  # Show the color scale for intensity values
plt.axis("off")  # Hide the axis for better visualization
plt.show()