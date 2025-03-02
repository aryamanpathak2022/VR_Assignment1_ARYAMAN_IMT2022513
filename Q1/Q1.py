
import os
import numpy as np
import cv2

# user input for paths
input_path = input("Enter the path to the input image: ")
output_path = input("Enter the directory to save the processed images: ")

# Ensures the output path is a directory
if not os.path.isdir(output_path):
    os.makedirs(output_path, exist_ok=True)

# Load the image
image = cv2.imread(input_path)
image2 = image  # Preserve original image

if image is None:
    raise ValueError("Error loading image. Check the file path.")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute scale factor to resize the image while maintaining aspect ratio
max_dim = max(image.shape[:2])
scale_factor = 1000 / max_dim if max_dim > 1000 else 1.0

# Resize both color and grayscale images
new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
image = cv2.resize(image, new_size)
gray = cv2.resize(gray, new_size)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Adaptive thresholding for edge detection
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
detected_circles = []

# Detect circular objects based on contour properties
for cnt in contours:
    perimeter = cv2.arcLength(cnt, True)  # Contour perimeter
    area = cv2.contourArea(cnt)  # Contour area

    if perimeter > 0:
        circular = (4 * np.pi * area) / (perimeter ** 2)  # Measure circularity

        # Check if the shape is roughly circular and meets the minimum area condition
        if 0.7 < circular < 1 and area > 450 * (scale_factor ** 2):
            detected_circles.append(cnt)

# Draw detected circular contours on a copy of the original image
processed_image = image.copy()
cv2.drawContours(processed_image, detected_circles, -1, (255, 0, 0), 2)

# Save the edges in the image
output_file = os.path.join(output_path, "edges_detected.jpg")
cv2.imwrite(output_file, processed_image)

# Create an empty mask for segmentation
mask = np.zeros_like(thresh, dtype=np.uint8)

# Draw filled contours (coins) on the mask
cv2.drawContours(mask, detected_circles, -1, 255, thickness=cv2.FILLED)

# Apply mask to the original image to extract coins
segmented = cv2.bitwise_and(image, image, mask=mask)

# Create a black background with the same shape as the image
processed_img = np.zeros_like(image, dtype=np.uint8)

# Overlay segmented coins on the black background
processed_img[mask == 255] = segmented[mask == 255]

# Save the segmented image
output_file = os.path.join(output_path, "segments.jpg")
cv2.imwrite(output_file, processed_img)

# Extract individual coin images
segmented_coins = []

for i, cnt in enumerate(detected_circles):
    # Find the minimum enclosing circle for each coin
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)

    # Create a circular mask for extraction
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(mask, center, radius, (255, 255, 255), -1)

    # Apply mask to extract the coin
    coin_segment = cv2.bitwise_and(image, mask)

    # Define cropping coordinates ensuring they stay within image bounds
    x1, y1 = max(center[0] - radius, 0), max(center[1] - radius, 0)
    x2, y2 = min(center[0] + radius, image.shape[1]), min(center[1] + radius, image.shape[0])

    # Crop the extracted coin area
    coin_segment = coin_segment[y1:y2, x1:x2]

    # Append to the list if the cropped image is not empty
    if coin_segment.size > 0:
        segmented_coins.append(coin_segment)

# Save the cropped images of individual coins
if segmented_coins:
    for idx, coin in enumerate(segmented_coins):
        output_file = os.path.join(output_path, f"coin_{idx}.jpg")
        cv2.imwrite(output_file, coin)
        print(f"Saved: {output_file}")
else:
    print("No coins detected or extracted.")

# Count and print the number of detected coins
coin_count = len(segmented_coins)
print(f'{coin_count} coins detected')

