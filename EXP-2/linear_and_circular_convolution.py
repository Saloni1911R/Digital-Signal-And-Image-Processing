import matplotlib.pyplot as plt
import numpy as np


def linear_convolution(signal1, signal2):
    return np.convolve(signal1, signal2, mode="full")


def circular_convolution(signal1, signal2):
    # Pad FFT length to (len1 + len2 - 1) to avoid aliasing and match linear convolution
    fft_length = len(signal1) + len(signal2) - 1

    fft_signal1 = np.fft.fft(signal1, fft_length)
    fft_signal2 = np.fft.fft(signal2, fft_length)

    # Compute IFFT and take real part
    circular_conv = np.real(np.fft.ifft(fft_signal1 * fft_signal2))
    return circular_conv


# Define discrete-time signals
signal1 = np.array([1, 2, 3, 4, 5])
signal2 = np.array([1, 0, 0, 1, 2])

# Compute convolutions
linear_conv = linear_convolution(signal1, signal2)
circular_conv = circular_convolution(signal1, signal2)

# Plot the linear and circular convolution results
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.stem(linear_conv)
plt.title("Linear Convolution")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.stem(circular_conv)
plt.title("Circular Convolution")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()