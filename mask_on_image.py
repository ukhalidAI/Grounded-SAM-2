import cv2
import numpy as np
from PIL import Image

# Load the original image and mask
original_image_path = "./testing/sample_image.jpg"  # Replace with your original image path
mask_image_path = "./testing/sample_mask.jpg"  # Replace with your mask image path

# Read images
original = cv2.imread(original_image_path)  # Load original image (BGR format)
mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale

# Ensure mask is binary (0 and 255)
_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Set the ROI in the original image to white where the mask is white
original[binary_mask == 255] = [255, 255, 255]  # White color in BGR format

# Save or display the result
result_path = "./testing/modified_image.png"
cv2.imwrite(result_path, original)

# Optionally display the result

