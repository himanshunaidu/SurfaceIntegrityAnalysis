"""
Performs affine transformations on images.
"""
import numpy as np
import cv2

PATH = 'img.png'
img = cv2.imread(PATH)

normalized_transform = [
    [0, -1, 1],
    [-1, 0, 1],
    [0, 0, 1]
]
h, w = img.shape[:2]
# Convert to pixel coordinates
pixel_transform = np.array(normalized_transform)
pixel_transform[0, 2] *= w / 2
pixel_transform[1, 2] *= h / 2
print("Pixel Transform before adjustment: \n", pixel_transform)

# Adjust for cv2's center of rotation
adjustment = np.array([
    [1, 0, -w / 2],
    [0, 1, -h / 2],
    [0, 0, 1]
])
pixel_transform = np.linalg.inv(adjustment) @ pixel_transform @ adjustment
print("Pixel Transform after adjustment: \n", pixel_transform)

# Apply transformation
img = cv2.warpAffine(img, pixel_transform[:2], (w, h))
cv2.imshow("Transformed", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# print("Image shape: ", img.shape)
# print("Pixel Transform: \n", pixel_transform)