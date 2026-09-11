import cv2
import numpy as np
import matplotlib.pyplot as plt

threshold_val = 165  
gamma_val = 1     

# Read image in grayscale
src_image = cv2.imread('/home/saloni/DSIP/EXP-7/Money.jpeg', cv2.IMREAD_GRAYSCALE)

if src_image is None:
    print("Image not found! Check file path.")
    exit()

# 1. Image Negation
negative_image = 255 - src_image

# 2. Thresholding
_, thresholded_image = cv2.threshold(src_image, threshold_val, 255, cv2.THRESH_BINARY)

# 3. Gamma Correction (Using Fast LUT)
invGamma = gamma_val
table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
gamma_corrected_image = cv2.LUT(src_image, table)

# Display 4-Panel Output
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(src_image, cmap='gray', vmin=0, vmax=255)
plt.title("Original Image", fontweight='bold')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(negative_image, cmap='gray', vmin=0, vmax=255)
plt.title("Negative Image", fontweight='bold')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(thresholded_image, cmap='gray', vmin=0, vmax=255)
plt.title(f"Threshold (T={threshold_val})", fontweight='bold')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(gamma_corrected_image, cmap='gray', vmin=0, vmax=255)
plt.title(f"Gamma (g={gamma_val})", fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()