import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read grayscale image
img = cv2.imread('imagess.jpeg', cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found. Please upload 'imagess.jpeg'.")

M, N = img.shape

# 2. Fourier Transform
F = np.fft.fftshift(np.fft.fft2(img))

# 3. Create frequency distance matrix D(u, v)
u, v = np.meshgrid(np.arange(N) - N // 2, np.arange(M) - M // 2)
D = np.sqrt(u**2 + v**2)

# 4. Gaussian Low-Pass and High-Pass Filter Design
D0 = 30  # Cutoff frequency radius

# Gaussian LPF formula: H(u,v) = exp(-D^2 / (2 * D0^2))
LPF = np.exp(-(D**2) / (2 * (D0**2)))
HPF = 1 - LPF

# 5. Apply filters and Inverse FFT
smooth = np.abs(np.fft.ifft2(np.fft.ifftshift(F * LPF)))
high_frequency_details = np.abs(np.fft.ifft2(np.fft.ifftshift(F * HPF)))

# High-frequency sharpening (Unsharp Masking style)
k = 1.2  # Sharpening factor
sharp = np.clip(img.astype(float) + k * high_frequency_details, 0, 255)

# 6. Display results
plt.figure(figsize=(14, 4))

plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(smooth, cmap='gray')
plt.title(f'Gaussian LPF (D0={D0})')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(high_frequency_details, cmap='gray')
plt.title('High-Pass Details (Edges)')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(sharp, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')

plt.tight_layout()
plt.show()
