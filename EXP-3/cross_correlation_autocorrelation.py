import numpy as np
import matplotlib.pyplot as plt

# Function to compute Cross-Correlation

def cross_correlation(signal1, signal2):
    return np.correlate(signal1, signal2, mode='full')

# Function to compute Auto-Correlation

def auto_correlation(signal):
    return np.correlate(signal, signal, mode='full')

# Define Discrete-Time Signals

signal1 = np.array([1, 2, 3, 4, 5])
signal2 = np.array([2, 4, 6, 8, 10])

# Compute Correlations
cross_corr = cross_correlation(signal1, signal2)
auto_corr = auto_correlation(signal1)

# Lag values
lags = np.arange(-(len(signal1)-1), len(signal2))

# Display Numerical Results

print("Signal 1 =", signal1)
print("Signal 2 =", signal2)

print("\nCross Correlation:")
print(cross_corr)

print("\nAuto Correlation:")
print(auto_corr)

# Plot Signals and Correlations

plt.figure(figsize=(12,8))

# Signal 1
plt.subplot(2,2,1)
plt.stem(range(len(signal1)), signal1)
plt.title("Signal 1")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid(True)

# Signal 2
plt.subplot(2,2,2)
plt.stem(range(len(signal2)), signal2)
plt.title("Signal 2")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid(True)

# Cross Correlation
plt.subplot(2,2,3)
plt.stem(lags, cross_corr)
plt.title("Cross-Correlation")
plt.xlabel("Lag")
plt.ylabel("Magnitude")
plt.grid(True)

# Auto Correlation
plt.subplot(2,2,4)
plt.stem(lags, auto_corr)
plt.title("Auto-Correlation")
plt.xlabel("Lag")
plt.ylabel("Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()