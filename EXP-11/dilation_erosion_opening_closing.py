import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the image in Grayscale mode
# Replace with your local or Colab image path
image_path = 'image.jpeg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}. Please upload the image.")

# 2. Define the Structuring Element (Kernel)
# A 5x5 square matrix filled with ones
kernel = np.ones((9, 9), np.uint8)

# 3. Perform Morphological Operations
erosion = cv2.erode(image, kernel, iterations=1)
dilation = cv2.dilate(image, kernel, iterations=1)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# 4. Plot and Display Results Side-by-Side
titles = ['Original Image', 'Erosion', 'Dilation', 'Opening', 'Closing']
images = [image, erosion, dilation, opening, closing]

plt.figure(figsize=(15, 8))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontsize=12)
    plt.axis('off')

plt.tight_layout()
plt.show()
