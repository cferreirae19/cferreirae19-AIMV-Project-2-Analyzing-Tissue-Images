import cv2 as cv
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score


# Load the image
image_file = input('Introduce a microscopy image: ')
if not os.path.isfile(image_file):
    print('The file ', image_file, 'does not exist')
    sys.exit(1)

image = cv.imread(image_file)
if image is None:
    print('The file ', image_file, "doesn't contain an image")
    sys.exit(1)

# Convert the image to YCbCr color space
ycbcr_image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)

# Define a range for tissue color in YCbCr color space
tissue_lower_bound = np.array([0, 133, 77])  # Adjust these values for Y, Cb, and Cr channels
tissue_upper_bound = np.array([255, 173, 127])  # Adjust these values for Y, Cb, and Cr channels

# Create a mask to identify tissue (nuclei and cytoplasm) in the YCbCr color space
tissue_mask = cv.inRange(ycbcr_image, tissue_lower_bound, tissue_upper_bound)

# Data initialization
data = ycbcr_image.reshape((-1, 3)).astype(np.float32)

# Optimal K value
optimal_k = 2

# Apply K-Means clustering to segment tissue from non-tissue areas
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv.kmeans(data, optimal_k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# Reshape labels to match the shape of the tissue mask
labels = labels.reshape(ycbcr_image.shape[:2])

# Create a blank nuclei mask with the same shape as the image
nuclei_mask = np.zeros(ycbcr_image.shape[:2], dtype=np.uint8)

# Update the nuclei mask based on clustering results
nuclei_mask[labels == 0] = 255

# Combine the tissue mask with the segmented nuclei mask
segmented_nuclei = cv.bitwise_and(tissue_mask, nuclei_mask)

# Create a binary stroma mask based on the initial tissue segmentation
stroma_mask = tissue_mask - segmented_nuclei

# Define the kernel for the closing operation
radius = 5  # Adjust this value as needed
closing_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1), (-1, -1))

# Apply morphological operations to ensure stroma regions are contiguous
stroma_mask = cv.morphologyEx(stroma_mask, cv.MORPH_CLOSE, closing_kernel)

# Find contours in the segmented nuclei mask
contours, _ = cv.findContours(segmented_nuclei, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Initialize a list to store the selected nuclei
selected_nuclei = []

# Variable initialization and image area calculations (in mm2)
nuclei_count = 0
image_area_mm2 = (ycbcr_image.shape[0] / 2000) * (ycbcr_image.shape[1] / 2000)

# Define the color wanted for the segmented nuclei
nuclei_color = (255, 255, 255)  # *REMEMBER: It's a BGR color!

# Define criteria for filtering contours
min_nucleus_area = 100  # Adjust this value as needed
max_nucleus_area = 2000  # Adjust this value as needed

# Create a copy of the original image to modify
modified_image = image.copy()  # !TESTING

# All the contours are checked
for contour in contours:
    # Get the bounding box of the nucleus contour
    x, y, w, h = cv.boundingRect(contour)
    # Check if the bounding box is entirely contained within the stroma mask
    if np.all(stroma_mask[y:y+h, x:x+w] == 0):
        # If not, add the nucleus contour to the list of selected nuclei
        selected_nuclei.append(contour)

    # Calculate the area of the contour
    area = cv.contourArea(contour)
    # Draw the contour on the modified image with black color
    cv.drawContours(modified_image, [contour], -1, (0, 0, 0), thickness=cv.FILLED)
    if min_nucleus_area < area < max_nucleus_area:
        # Increase the nuclei count if it has an area between the established boundaries
        nuclei_count += 1

# Draw the selected nuclei on the modified image
cv.drawContours(modified_image, selected_nuclei, -1, nuclei_color, thickness=cv.FILLED)

# Calculate cell density for the selected nuclei
nuclei_count = len(selected_nuclei)

# Calculate cell density (cells per mm2)
cell_density = nuclei_count / image_area_mm2

# Display the number of nuclei and cell density in the console
print("Number of Nuclei:", nuclei_count)
print("Cell Density (cells/mm²):", cell_density)

# Show the segmented nuclei mask and the modified image (resizable windows)
cv.namedWindow("Segmented Nuclei Mask", cv.WINDOW_NORMAL)
cv.imshow("Segmented Nuclei Mask", segmented_nuclei)
cv.namedWindow("Modified Image", cv.WINDOW_NORMAL)
cv.imshow("Modified Image", modified_image)
cv.waitKey(0)
cv.destroyAllWindows()
