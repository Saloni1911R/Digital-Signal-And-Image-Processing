import numpy as np

import matplotlib.pyplot as plt

signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Compute the FFT of the signal

fft_result = np.fft.fft(signal)

# Compute the magnitude and Phase spectrum of the FFT result

magnitude_spectrum = np.abs(fft_result)

Phase_spectrum = np.angle(fft_result)

reconstructed_signal = np.fft.ifft(fft_result)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.stem(signal)
plt.title("Original Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.stem(magnitude_spectrum)
plt.title("Magnitude Spectrum")
plt.xlabel("Frequency Bin")
plt.ylabel("Magnitude")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.stem(Phase_spectrum)
plt.title("Phase Spectrum (Radians)")
plt.xlabel("Frequency Bin")
plt.ylabel("Phase")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.stem(np.real(reconstructed_signal))
plt.title("Reconstructed Signal (IFFT)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()