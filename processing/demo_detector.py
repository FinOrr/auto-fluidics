import cv2
import numpy as np
import os
import time

# Get the list of input files in the directory
files = os.listdir('img/')

# Filter the list to keep only the '.png' files
png_files = sorted([file for file in files if file.endswith('.png')])

# Create an empty dictionary to store images
images = {}

# Create a dictionary to store metrics for each image
image_metrics = {}

# Load each image into the dictionary
for filename in png_files:
    # Construct the full path to the image file
    filepath = os.path.join('img/', filename)
    
    # Read the image using OpenCV
    image = cv2.imread(filepath)
    
    # Store the image in the dictionary
    images[filename] = image

sample = images['sample-1.png']

# Convert to grayscale
grayscale = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

# Define the motion kernel for deblurring (assuming motion in the x-axis)
kernel_size = 9  # Adjust the kernel size according to the blur extent
motion_kernel = np.zeros((kernel_size, kernel_size))
motion_kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
motion_kernel /= kernel_size

# Perform Wiener deconvolution for motion deblurring
preprocessed_image = cv2.filter2D(grayscale, -1, motion_kernel)

# Apply noise reduction using Fast Non-local Means Denoising
# preprocessed_image = cv2.fastNlMeansDenoising(preprocessed_image, None, h=10, templateWindowSize=30, searchWindowSize=30)

# Equalise and threshold
preprocessed_image = cv2.equalizeHist(preprocessed_image)
_, preprocessed_image = cv2.threshold(preprocessed_image, 24, 255, cv2.THRESH_BINARY)

# Apply Sobel edge detection
sobel_x = cv2.Sobel(preprocessed_image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(preprocessed_image, cv2.CV_64F, 0, 1, ksize=5)

# Combine the gradient images to obtain the magnitude
sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

# Normalise the Sobel edge magnitude image
sobel_magnitude_normalized = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Threshold the Sobel edge magnitude image to obtain a binary edge image
_, preprocessed_image = cv2.threshold(sobel_magnitude_normalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Find contours in the preprocessed image
contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Init lists to store circle information
circle_metrics = []

# Filter out contours that aren't circles based on circularity
for contour in contours:
    # Calculate contour area and perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Calculate circularity
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    
    # Set a circularity threshold
    circularity_threshold = 0.8  # Adjust as needed
    
    # If the contour is sufficiently circular, fit a circle to it
    if circularity > circularity_threshold:
        # Fit a circle to the contour using the method of least squares
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Append circle metrics to the list
        circle_metrics.append({
            'mean_radius': None,
            'std_dev_radius': None,
            'radius': radius,
            'center': center,
            'circularity': circularity,
            'area': area,
            'perimeter': perimeter
        })

    # Calculate mean and standard deviation of circle radii
    circle_radii = [metric['radius'] for metric in circle_metrics]
    mean_radius = np.mean(circle_radii)
    std_dev_radius = np.std(circle_radii)

    # Calculate the threshold for omitting circles
    min_threshold_radius = mean_radius - (1.5 * std_dev_radius)
    max_threshold_radius = mean_radius + (1.5 * std_dev_radius)

    # Filter out circles beyond the threshold
    filtered_circle_metrics = [metric for metric in circle_metrics if min_threshold_radius <= metric['radius'] <= max_threshold_radius]

    # Draw the filtered circles on the original image
    for metric in filtered_circle_metrics:
        cv2.circle(sample, metric['center'], metric['radius'], (0, 255, 0), 2)

cv2.imshow('Annotated Image', sample)
cv2.waitKey(0)
cv2.destroyAllWindows()
