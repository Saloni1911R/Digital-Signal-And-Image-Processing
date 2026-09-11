import cv2
import numpy as np
import matplotlib.pyplot as plt

image_paths = [
    '/home/saloni/DSIP/EXP-7/image1.png',
    '/home/saloni/DSIP/EXP-7/image2.png',
    '/home/saloni/DSIP/EXP-7/image3.png',
    '/home/saloni/DSIP/EXP-7/image4.png',
    '/home/saloni/DSIP/EXP-7/image5.png'
]


images = []

for path in image_paths:

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Image not found:", path)
        exit()

    images.append(image)


def show_images(threshold_val, gamma_val):

    plt.figure(figsize=(20, 8))

    for i in range(5):

        src_image = images[i]


        negative_image = 255 - src_image

        _, thresholded_image = cv2.threshold(
            src_image,
            threshold_val,
            255,
            cv2.THRESH_BINARY
        )

        table = np.array([
            ((j / 255.0) ** gamma_val) * 255
            for j in np.arange(0, 256)
        ]).astype("uint8")

        gamma_image = cv2.LUT(
            src_image,
            table
        )


        # We display the original image here.
        # Change "src_image" to negative_image,
        # thresholded_image or gamma_image if required.

        plt.subplot(1, 5, i + 1)

        plt.imshow(
            src_image,
            cmap='gray',
            vmin=0,
            vmax=255
        )

        plt.title(
            f"Image {i + 1}",
            fontweight='bold'
        )

        plt.axis('off')


    plt.suptitle(
        f"All 5 Images | Threshold = {threshold_val} | Gamma = {gamma_val}",
        fontsize=16,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.show()


values = [
    (50, 0.5),
    (100, 0.5),
    (128,0.8),
    (150, 0.5),
    (165, 0.5),
    (200, 0.5),

    (165, 0.8),
    (165, 1.0),
    (165, 1.5),
    (165, 2.0)
]



for threshold_val, gamma_val in values:

    print(
        f"Threshold = {threshold_val}, "
        f"Gamma = {gamma_val}"
    )

    show_images(
        threshold_val,
        gamma_val
    )