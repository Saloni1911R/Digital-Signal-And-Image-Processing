import cv2
import matplotlib.pyplot as plt

image_paths = [
    '/home/saloni/DSIP/EXP-8/image1.png',
    '/home/saloni/DSIP/EXP-8/image2.png',
    '/home/saloni/DSIP/EXP-8/image3.png',
    '/home/saloni/DSIP/EXP-8/image4.png',
    '/home/saloni/DSIP/EXP-8/image5.png'
]

for image_path in image_paths:

    # 1. Load grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not open: {image_path}")

    # 2. Histogram equalization
    equalized_image = cv2.equalizeHist(image)

    # 3. Calculate histograms
    hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

    # 4. Display images
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Equalized Image')
    plt.imshow(equalized_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 5. Display histograms
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(hist_original)
    plt.xlim([0, 256])
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title('Equalized Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(hist_equalized)
    plt.xlim([0, 256])
    plt.grid(True)

    plt.tight_layout()
    plt.show()