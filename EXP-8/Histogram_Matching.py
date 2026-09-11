import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. Load the source and reference images
source_path = '/home/saloni/DSIP/EXP-8/IMG_20230915_135931789.jpg'
reference_path = '/home/saloni/DSIP/EXP-8/anime1-255643.webp'

source_image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
reference_image = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)

# Error safety check
if source_image is None or reference_image is None:
    raise FileNotFoundError("Check your image paths; one of the source or reference files failed to load.")

# 2. Calculate histograms
source_hist = cv2.calcHist([source_image], [0], None, [256], [0, 256])
reference_hist = cv2.calcHist([reference_image], [0], None, [256], [0, 256])

# 3. Normalize histograms (calculate probability density function)
source_hist /= source_hist.sum()
reference_hist /= reference_hist.sum()

# 4. Calculate Cumulative Distribution Functions (CDF)
source_cdf = source_hist.cumsum()
reference_cdf = reference_hist.cumsum()

# 5. Create Look-Up Table (LUT) mapping
# Create an array of all possible pixel intensities (0-255)
pixel_values = np.arange(256)
# Interpolate source CDF values over the reference profile
lut_mapping = np.interp(source_cdf, reference_cdf, pixel_values).astype(np.uint8)

# 6. Apply the Look-Up Table safely and rapidly
matched_image = cv2.LUT(source_image, lut_mapping)

# 7. Display the images using Matplotlib
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.title('Source Image')
plt.imshow(source_image, cmap='gray')
plt.axis('off')

plt.subplot(132)
plt.title('Reference Image')
plt.imshow(reference_image, cmap='gray')
plt.axis('off')

plt.subplot(133)
plt.title('Matched Image')
plt.imshow(matched_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
