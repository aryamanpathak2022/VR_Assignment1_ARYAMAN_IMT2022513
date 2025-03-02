# Coin Detection, Segmentation, and Counting

## Overview
This project uses computer vision techniques to detect, segment, and count coins from an image containing scattered Indian coins. The implementation is done in a Jupyter Notebook (`.ipynb`).

## Requirements
To run the notebook, install the required dependencies using:
```bash
pip install opencv-python numpy
```

## Steps

### 1. Preprocessing the Image
- Convert the image to grayscale.
- Resize while maintaining aspect ratio.
- Apply Gaussian blur.
- Use adaptive thresholding for better edge detection.

### 2. Coin Detection
- Find contours in the thresholded image.
- Filter out circular objects based on perimeter and area.
- Visualize detected coins by outlining them in green.

### 3. Coin Segmentation
- Create a binary mask based on detected contours.
- Extract segmented coin images on a black background.
- Save the segmented image.

### 4. Individual Coin Extraction
- Extract each detected coin using minimum enclosing circles.
- Save each coin separately.

### 5. Coin Counting
- Count the total number of detected coins.
- Display the count as an output.

## Running the Notebook
1. Open the Jupyter Notebook (`.ipynb` file) in Google Colab or JupyterLab.
2. Upload an image containing scattered Indian coins.
3. Run the cells sequentially to process the image and obtain results.

## Expected Outputs
- `edges_detected.jpg`: Image with detected coin contours outlined.
- `segments.jpg`: Image with segmented coins on a black background.
- `coin_X.jpg`: Individual extracted coins.
- Console Output: Total number of detected coins.


## Sample Input 

### 1. Original Image (Sample Input)

![Sample Input Image](images/sample_input.jpg)

This is the original image that will be processed.


---

## Sample Output

### 1. Grayscale Conversion (Expected Output)

![Grayscale Image](images/grayscale_output.jpg)

The original image is converted into grayscale.

### 2. Edge Detection (Expected Output)

![Edge Detection](images/edge_output.jpg)

Edges of objects in the image are detected using Canny edge detection.

### 3. Image Segmentation (Expected Output)

![Segmented Image](images/segmentation_output.jpg)

The image is segmented based on intensity values.

### 4. Number of coins (Expected Console Output)

```
6 coins detected
```

Ensure that all input images are placed in the correct directory before running the notebook.


# Image Stitching using OpenCV

## Overview
This project performs image stitching using SIFT feature detection and homography transformation to align and merge images into a panoramic view. It reads a sequence of images, detects keypoints, matches them, computes a homography matrix, and stitches images together.


## Requirements

To run the notebook, install the required dependencies using:
```bash

pip install numpy opencv-python imutils tqdm
```

## Steps

### 🔹 1. Read & Sort Images
- The script reads images from the input directory.
- It filters out non-numeric filenames and sorts them in numerical order.

```python
img_path = [os.path.join(input_dir, i) for i in os.listdir(input_dir) if os.path.splitext(i)[0].isdigit()]
img_path.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
```

### 🔹 2. Load the First Image
- Reads and resizes the first image to 600px width for consistency.

```python
left_img = cv2.imread(img_path[0])
left_img = imutils.resize(left_img, width=600)
cv2_imshow(left_img)
```

### 🔹 3. Detect Keypoints & Descriptors using SIFT
- Uses the SIFT algorithm to extract feature points from each image.

```python
descriptor = cv2.SIFT_create()
kpsA, desA = descriptor.detectAndCompute(left_img, None)
kpsB, desB = descriptor.detectAndCompute(right_img, None)
```

### 🔹 4. Match Keypoints using KNN & Apply Ratio Test
- Uses K-Nearest Neighbors (KNN) to match features between consecutive images.
- Applies Lowe’s ratio test (0.75 threshold) to filter good matches.

```python
matcher = cv2.BFMatcher()
rawMatches = matcher.knnMatch(desA, desB, k=2)

matches = []
for m in rawMatches:
    if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
        matches.append((m[0].trainIdx, m[0].queryIdx))
```

### 🔹 5. Compute Homography & Warp Image
- Homography matrix aligns one image onto the next.
- Uses cv2.RANSAC with a threshold of 5.0 for robust transformation.

```python
H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 5.0)
pano_img = cv2.warpPerspective(left_img, H, (new_width, new_height))
```

### 🔹 6. Blend & Stitch Images
- The right image is blended onto the warped perspective of the left image.
- Crops excess black areas using contours & bounding box extraction.

```python
gray = cv2.cvtColor(pano_img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    x, y, w, h = cv2.boundingRect(contours[0])
    pano_img = pano_img[y:y+h-1, x:x+w-1]
```

---

## 📌 Expected Output

| Sample Input | Expected Output |
|-------------|----------------|
| Left Image  | Stitched Image  |

---

## 📥 Sample Input

Sample directory structure:

```
input_dir/
 ├── 1.jpg
 ├── 2.jpg
 ├── 3.jpg
```

---

## 📤 Sample Output

Final stitched panorama saved in:

```
output_dir/
 ├── pano3.jpg
```

