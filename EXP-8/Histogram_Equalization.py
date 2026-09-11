import cv2
import matplotlib.pyplot as plt

# 1. Load the grayscale image
image_path = '/home/saloni/DSIP/EXP-8/images.jpeg' 
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image loaded correctly
if image is None:
    raise FileNotFoundError(f"Could not open or find the image at {image_path}")

# 2. Perform histogram equalization
equalized_image = cv2.equalizeHist(image)

# 3. Calculate histograms
hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

# --- FIGURE 1: Visual Image Comparison ---
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Equalized Image')
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

plt.tight_layout()

# --- FIGURE 2: Histogram Comparison ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title('Original Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.plot(hist_original, color='crimson')
plt.xlim([0, 256])
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title('Equalized Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.plot(hist_equalized, color='royalblue')
plt.xlim([0, 256])
plt.grid(True)

plt.tight_layout()
plt.show()
