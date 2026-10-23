import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Create a clean synthetic binary image with distinct geometric shapes
binary_image = np.zeros((300, 300), dtype=np.uint8)

# Shape A: Solid White Square (100x100)
binary_image[30:130, 30:130] = 255

# Shape B: Cross/Plus Pattern (+)
cv2.line(binary_image, (200, 50), (200, 110), 255, 3)
cv2.line(binary_image, (170, 80), (230, 80), 255, 3)

# Shape C: Isolated Single Noise Pixels
binary_image[200, 50] = 255
binary_image[220, 80] = 255

# Shape D: Isolated Horizontal Line Segment
binary_image[250, 180:210] = 255

# -------------------------------------------------------------
# DEFINE STRUCTURING ELEMENTS
# -------------------------------------------------------------
sample_kernels = {
    "Sample 1: Cross/Plus Pattern (+)": np.array([
        [-1,  1, -1],
        [ 1,  1,  1],
        [-1,  1, -1]
    ], dtype=np.int8),

    "Sample 2: Isolated Pixel Detection": np.array([
        [-1, -1, -1],
        [-1,  1, -1],
        [-1, -1, -1]
    ], dtype=np.int8),

    "Sample 3: Top-Left Corner Detector": np.array([
        [-1, -1, -1],
        [-1,  1,  1],
        [-1,  1,  1]
    ], dtype=np.int8),

    "Sample 4: Horizontal Line Segment": np.array([
        [-1, -1, -1],
        [ 1,  1,  1],
        [-1, -1, -1]
    ], dtype=np.int8)
}

# Display Original Input
plt.figure(figsize=(5, 5))
plt.imshow(binary_image, cmap='gray')
plt.title("Synthetic Binary Test Image", fontsize=12)
plt.axis('off')
plt.show()

# Display Results Separately
for title, kernel in sample_kernels.items():
    result = cv2.morphologyEx(binary_image, cv2.MORPH_HITMISS, kernel)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(result, cmap='gray')
    plt.title(title, fontsize=12)
    plt.axis('off')
    plt.show()
