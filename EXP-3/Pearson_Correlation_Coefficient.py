import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

print("Loading audio files...")

original, sr = librosa.load("music orignal.mp3", sr=22050, mono=True)
karaoke, sr = librosa.load("background music.mp3", sr=22050, mono=True)
different, sr = librosa.load("other song.mp3", sr=22050, mono=True)

# Make all signals same length

length = min(len(original), len(karaoke), len(different))

original = original[:length]
karaoke = karaoke[:length]
different = different[:length]

print("Audio loaded successfully!")
print(f"Sampling Rate : {sr} Hz")
print(f"Samples Used  : {length}")

# Pearson Correlation

corr_ok = np.corrcoef(original, karaoke)[0, 1]
corr_od = np.corrcoef(original, different)[0, 1]
corr_kd = np.corrcoef(karaoke, different)[0, 1]

print(f"Original vs Karaoke   : {corr_ok:.4f}")
print(f"Original vs Different : {corr_od:.4f}")
print(f"Karaoke vs Different  : {corr_kd:.4f}")

# Cross Correlation

cross = correlate(original, karaoke, mode='full')

# Normalize
cross = cross / (np.linalg.norm(original) * np.linalg.norm(karaoke))

print(f"\nMaximum Cross Correlation : {np.max(cross):.4f}")


# Display Waveforms (First 5 Seconds)

duration = 5
samples = min(sr * duration, length)

time = np.arange(samples) / sr

plt.figure(figsize=(14, 9))

plt.subplot(3,1,1)
plt.plot(time, original[:samples], color='blue')
plt.title("Original Song Signal")
plt.xlabel("Time (Seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(time, karaoke[:samples], color='green')
plt.title("Karaoke Song Signal")
plt.xlabel("Time (Seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(time, different[:samples], color='red')
plt.title("Different Song Signal")
plt.xlabel("Time (Seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()

# Display Cross Correlation

lags = np.arange(-length + 1, length)

plt.figure(figsize=(14,5))
plt.plot(lags, cross, color='purple')
plt.title("Cross Correlation (Original vs Karaoke)")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.grid(True)

plt.tight_layout()

plt.show()

print("\nAnalysis Complete!")