import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read grayscale image
img = cv2.imread('/content/imagess.jpeg', cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Please upload 'imagess.jpeg' to /content/ directory.")

M, N = img.shape

# 2. Fourier Transform
F = np.fft.fftshift(np.fft.fft2(img))

# 3. Distance Matrix
u, v = np.meshgrid(np.arange(N) - N // 2, np.arange(M) - M // 2)
D = np.sqrt(u**2 + v**2)

# 4. Experiment with multiple cutoff frequencies (D0 values)
cutoff_values = [10, 30, 80]

fig, axes = plt.subplots(len(cutoff_values), 3, figsize=(12, 10))

for idx, D0 in enumerate(cutoff_values):
    # Gaussian Low-Pass and High-Pass Filters
    LPF = np.exp(-(D**2) / (2 * (D0**2)))
    HPF = 1 - LPF

    # Filter application
    smooth = np.abs(np.fft.ifft2(np.fft.ifftshift(F * LPF)))
    high_freq = np.abs(np.fft.ifft2(np.fft.ifftshift(F * HPF)))
    sharp = np.clip(img.astype(float) + 1.2 * high_freq, 0, 255)

    # Plot Low-Pass (Blur)
    axes[idx, 0].imshow(smooth, cmap='gray')
    axes[idx, 0].set_title(f'LPF (Smooth) - D0 = {D0}')
    axes[idx, 0].axis('off')

    # Plot High-Pass (Edges)
    axes[idx, 1].imshow(high_freq, cmap='gray')
    axes[idx, 1].set_title(f'HPF (Edges) - D0 = {D0}')
    axes[idx, 1].axis('off')

    # Plot Sharpened Image
    axes[idx, 2].imshow(sharp, cmap='gray')
    axes[idx, 2].set_title(f'Sharpened - D0 = {D0}')
    axes[idx, 2].axis('off')

plt.tight_layout()
plt.show()
