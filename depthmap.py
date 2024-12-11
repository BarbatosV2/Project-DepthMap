import cv2
import numpy as np

# Load left and right images
img_left = cv2.imread('left_image.png', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('right_image.png', cv2.IMREAD_GRAYSCALE)

# Check if images are loaded
if img_left is None or img_right is None:
    raise IOError("Error loading images. Ensure the paths are correct.")

# Resize images to fit the screen (maintain aspect ratio)
max_width = 800  # Maximum width to fit on the screen
scale_factor = max_width / max(img_left.shape[1], img_right.shape[1])  # Calculate scaling factor

new_width = int(img_left.shape[1] * scale_factor)
new_height = int(img_left.shape[0] * scale_factor)

img_left_resized = cv2.resize(img_left, (new_width, new_height), interpolation=cv2.INTER_AREA)
img_right_resized = cv2.resize(img_right, (new_width, new_height), interpolation=cv2.INTER_AREA)

# Parameters for StereoSGBM
min_disparity = 0
num_disparities = 16 * 5  # Must be divisible by 16
block_size = 7
p1 = 8 * 3 * block_size ** 2  # Control the disparity smoothness
p2 = 32 * 3 * block_size ** 2
disp12_max_diff = 1
uniqueness_ratio = 15
speckle_window_size = 0
speckle_range = 2

# Create a StereoSGBM object
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disparity,
    numDisparities=num_disparities,
    blockSize=block_size,
    P1=p1,
    P2=p2,
    disp12MaxDiff=disp12_max_diff,
    uniquenessRatio=uniqueness_ratio,
    speckleWindowSize=speckle_window_size,
    speckleRange=speckle_range
)

# Compute disparity map
disparity = stereo.compute(img_left_resized, img_right_resized)

# Normalize the disparity map for better visualization
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)

# Display the disparity map
cv2.imshow("Disparity Map (SGBM)", disparity_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
