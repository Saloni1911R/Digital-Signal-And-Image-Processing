import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def fft_lowpass_filter(image, d0=40, n=2):
    """Frequency-domain Butterworth Lowpass Filter for noise reduction."""
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(v - ccol, u - crow)
    D = np.sqrt(U**2 + V**2)
    
    H = 1 / (1 + (D / (d0 + 1e-5))**(2 * n))
    
    fshift_filtered = fshift * H
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return np.uint8(np.clip(img_back, 0, 255))

def fft_highboost_filter(image, d0=30, k=1.8):
    """Frequency-domain High-Boost Filter for detail enhancement."""
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(v - ccol, u - crow)
    D = np.sqrt(U**2 + V**2)
    
    n = 2
    H_lp = 1 / (1 + (D / (d0 + 1e-5))**(2 * n))
    H_hp = 1 - H_lp
    
    H_hb = (k - 1) + H_hp
    
    fshift_filtered = fshift * H_hb
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return np.uint8(np.clip(img_back, 0, 255))

def extract_boundaries(image):
    """Extract boundary maps using bilateral filter and Canny edge detection."""
    blurred = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    return edges

# List of files
image_files = [
    "Pasted image.png",
    "Pasted image (2).png",
    "Pasted image (3).png",
    "Pasted image (4).png",
    "Pasted image (5).png"
]

output_folder = "separate_outputs"
os.makedirs(output_folder, exist_ok=True)

for img_name in image_files:
    if not os.path.exists(img_name):
        print(f"Skipping (File not found): {img_name}")
        continue
        
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    base_name = os.path.splitext(img_name)[0]
    
    # Process pipeline
    smoothed = fft_lowpass_filter(img, d0=40, n=2)
    sharpened = fft_highboost_filter(img, d0=30, k=1.8)
    boundaries = extract_boundaries(img)
    
    # Dictionary of outputs to process separately
    results = {
        "Original": img,
        "FFT_Lowpass_Smoothed": smoothed,
        "FFT_HighBoost_Sharpened": sharpened,
        "Boundary_Extraction": boundaries
    }
    
    # Display and Save EACH image separately
    for label, output_img in results.items():
        # Save as individual image file
        save_path = os.path.join(output_folder, f"{base_name}_{label}.png")
        cv2.imwrite(save_path, output_img)
        
        # Display in a separate Matplotlib window
        plt.figure(figsize=(6, 6))
        plt.imshow(output_img, cmap='gray')
        plt.title(f"{base_name} - {label}")
        plt.axis('off')
        plt.show()

print(f"All images processed and saved separately in folder: '{output_folder}'")