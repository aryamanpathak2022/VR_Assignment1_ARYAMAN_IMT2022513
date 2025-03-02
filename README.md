# Coin Detection, Segmentation, and Counting

## Overview
This project uses computer vision techniques to detect, segment, and count coins from an image containing scattered Indian coins. The implementation is done in a Jupyter Notebook (`.ipynb`).

## Requirements
To run the notebook, install the required dependencies using:
```bash
pip install opencv-python numpy argparse
```

## Steps


###  1. User Input
- The script first asks user for input image path  and output directory to save the processed images:

### 2. Preprocessing the Image
- Convert the image to grayscale.
- Resize while maintaining aspect ratio (max_dim == 1000).
- Apply Gaussian blur (7x7).
- Uses adaptive thresholding for better edge detection.

### 3. Coin Detection
- Find contours in the thresholded image.
- Filter out circular objects based on perimeter and area.
- Visualize detected coins by outlining them in blue.

### 4. Coin Segmentation
- Create a binary mask based on detected contours.
- Extract segmented coin images on a black background.
- Save the segmented image.

### 5. Individual Coin Extraction
- Extract each detected coin using minimum enclosing circles.
- Crop and save each coin separately.

### 6. Coin Counting
- Counts the total number of detected coins.
- Displays the count as an console output.

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

![Sample Input Image](Q1/data/2.png)

This is the original image that will be processed.


---

## Sample Output

### 1. Edge Detection (Expected Output)

![Edge Detection](Q1/data/2_output/edges_detected.jpg)

Edges of objects in the image are detected using Canny edge detection.

### 2. Image Segmentation (Expected Output)

![Segmented Image](Q1/data/2_output/segments.jpg)

The image is segmented based on intensity values.

### 2. Individual Coins Cropped   (Expected Output)

![coin Image](Q1/data/2_output/coin_0.jpg)
![coin Image](Q1/data/2_output/coin_1.jpg)

![coin Image](Q1/data/2_output/coin_2.jpg)

![coin Image](Q1/data/2_output/coin_3.jpg)

![coin Image](Q1/data/2_output/coin_4.jpg)

![coin Image](Q1/data/2_output/coin_5.jpg)

![coin Image](Q1/data/2_output/coin_6.jpg)


The individual images for each coins detected

### 4. Number of coins (Expected Console Output)

```
7 coins detected
```

![Segmented Image](Q1/data/2_output/console.png)
Ensure that all input images are placed in the correct directory before running the notebook.



# Image Stitching using OpenCV

## Overview
This project performs image stitching using SIFT feature detection and homography transformation to align and merge images into a panoramic view. It reads a sequence of images, detects keypoints, matches them, computes a homography matrix, and stitches images together.


## Requirements

To run the notebook, install the required dependencies using:
```bash

pip install numpy opencv-python imutils tqdm argparse
```

## Steps

###  1. User Input
- The script first asks user for input and output directories path
- The input directory must contain images named in sequential numerical order, such as 1.png, 2.png, …, N.png. The output directory will store the generated panorama image, which will be saved as panorama.jpg.

###  2. Read & Sort Images
- The script reads images from the input directory.
- It filters out non-numeric filenames and sorts them in numerical order.

###  3. Load the First Image
- Reads and resizes the first image to 600px width for consistency.


###  4. Detect Keypoints & Descriptors using SIFT
- Uses the SIFT algorithm to extract feature points from each image by looping over the input images.


### 5. Match Keypoints using KNN & Apply Ratio Test
- Using K-Nearest Neighbors (KNN) to match features between consecutive images.
- Applies Lowe’s ratio test (0.75 threshold) to filter good matches.


### 6. Compute Homography & Warp Image
- Homography matrix aligns one image onto the next.
- Uses cv2.RANSAC with a threshold of 5.0 for  transformation.


### 7. Blend & Stitch Images
- The right image is blended onto the warped perspective of the left image.
- Crops excess black areas using contours & bounding box extraction.

### 8. Repeat
- The combined images becomes the left image now and the next image will be merged repating the above steps

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
### 1. Sample Input Images

![Sample Input Image](Q2/data/example_1/1.png)
![Sample Input Image](Q2/data/example_1/2.png)
![Sample Input Image](Q2/data/example_1/3.png)

---

## 📤 Sample Output

Final stitched panorama saved in:

```
output_dir/
 ├── panorama.jpg
```

### 1. Sample Output Image
![Sample Output Image](Q2/data/example_1_output/panorama.jpg)

