import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. Source images
source_paths = [
    '/home/saloni/DSIP/EXP-8/image1.png',
    '/home/saloni/DSIP/EXP-8/image2.png',
    '/home/saloni/DSIP/EXP-8/image3.png',
    '/home/saloni/DSIP/EXP-8/image4.png',
    '/home/saloni/DSIP/EXP-8/image5.png'
]

# Reference image
reference_path = '/home/saloni/DSIP/EXP-8/Money.jpeg'

reference_image = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)

if reference_image is None:
    raise FileNotFoundError("Reference image failed to load.")

# Reference histogram
reference_hist = cv2.calcHist(
    [reference_image], [0], None, [256], [0, 256]
)

reference_hist /= reference_hist.sum()

# Reference CDF
reference_cdf = reference_hist.cumsum()

# Pixel values
pixel_values = np.arange(256)

# 2. Process all 5 source images
for source_path in source_paths:

    source_image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)

    if source_image is None:
        raise FileNotFoundError(f"Source image failed to load: {source_path}")

    # Source histogram
    source_hist = cv2.calcHist(
        [source_image], [0], None, [256], [0, 256]
    )

    source_hist /= source_hist.sum()

    # Source CDF
    source_cdf = source_hist.cumsum()

    # Create LUT
    lut_mapping = np.interp(
        source_cdf,
        reference_cdf,
        pixel_values
    ).astype(np.uint8)

    # Histogram matching
    matched_image = cv2.LUT(source_image, lut_mapping)

    # 3. Display output
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Source Image')
    plt.imshow(source_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Reference Image')
    plt.imshow(reference_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Matched Image')
    plt.imshow(matched_image, cmap='gray')
    plt.axis('off')

    plt.suptitle(source_path.split('/')[-1])
    plt.tight_layout()
    plt.show()