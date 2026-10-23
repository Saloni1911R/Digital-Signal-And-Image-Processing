import matplotlib
matplotlib.use('TkAgg')  # Switched to TkAgg to bypass the GLIBC library conflict
import matplotlib.pyplot as plt
import numpy as np

# 1. Define the signal with distinct real and imaginary values
signal = np.array(
    [1 + 2j, 2 - 1j, 3 + 0j, 4 + 3j, 5 - 2j, 6 + 1j, 7 - 3j, 8 + 4j]
)

# Compute the FFT of the signal
fft_result = np.fft.fft(signal)

# Compute the magnitude and Phase spectrum of the FFT result
magnitude_spectrum = np.abs(fft_result)
Phase_spectrum = np.angle(fft_result)

# Compute the IFFT of the FFT result
reconstructed_signal = np.fft.ifft(fft_result)

# Plotting the 2x2 grid structured with value tracking
plt.figure(figsize=(14, 10))

# 1. Plotting the raw FFT result (Real and Imaginary data points)
plt.subplot(2, 2, 1)
real_parts = np.real(fft_result)
imag_parts = np.imag(fft_result)
plt.stem(real_parts, linefmt="b-", markerfmt="bo", label="Real Part")
plt.stem(imag_parts, linefmt="r--", markerfmt="rx", label="Imag Part")

# Add text values for Real and Imaginary points
for i, (r_val, i_val) in enumerate(zip(real_parts, imag_parts)):
    plt.text(
        i, r_val + 0.5 if r_val >= 0 else r_val - 1.5, f"{r_val:.1f}", 
        color="blue", ha="center", va="center", fontsize=9
    )
    plt.text(
        i, i_val + 0.5 if i_val >= 0 else i_val - 1.5, f"{i_val:.1f}j", 
        color="red", ha="center", va="center", fontsize=9
    )

plt.title("Raw FFT Datapoints")
plt.xlabel("Frequency Bin")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

# 2. Magnitude Spectrum
plt.subplot(2, 2, 2)
plt.stem(magnitude_spectrum)

# Add text values for Magnitude points
for i, mag in enumerate(magnitude_spectrum):
    plt.text(
        i, mag + 0.8, f"{mag:.1f}", 
        color="black", ha="center", va="bottom", fontsize=9
    )

plt.title("Magnitude Spectrum")
plt.xlabel("Frequency Bin")
plt.ylabel("Magnitude")
plt.grid(True)

# 3. Phase Spectrum
plt.subplot(2, 2, 3)
plt.stem(Phase_spectrum)

# Add text values for Phase points
for i, phase in enumerate(Phase_spectrum):
    plt.text(
        i, phase + 0.1 if phase >= 0 else phase - 0.2, f"{phase:.2f}", 
        color="black", ha="center", va="center", fontsize=9
    )

plt.title("Phase Spectrum (Radians)")
plt.xlabel("Frequency Bin")
plt.ylabel("Phase")
plt.grid(True)

# 4. Reconstructed Signal (IFFT)
plt.subplot(2, 2, 4)
recon_real = np.real(reconstructed_signal)
plt.stem(recon_real)

# Add text values for Reconstructed points
for i, recon in enumerate(recon_real):
    plt.text(
        i, recon + 0.2, f"{recon:.1f}", 
        color="black", ha="center", va="bottom", fontsize=9
    )

plt.title("Reconstructed Signal (IFFT)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()
