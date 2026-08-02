import librosa
import numpy as np
import matplotlib.pyplot as plt


original, sr = librosa.load("music orignal.mp3", sr=22050, mono=True)
karaoke, sr = librosa.load("background music.mp3", sr=22050, mono=True)
different, sr = librosa.load("other song.mp3", sr=22050, mono=True)

length = min(len(original), len(karaoke), len(different))
original = original[:length]
karaoke = karaoke[:length]
different = different[:length]


# --- 1. Calculate Signal Differences ---
diff_orig_karaoke = original - karaoke  # Difference 1
diff_orig_diff = original - different     # Difference 2

# --- 2. Calculate Energy & Power Statistics ---
energy_diff_ok = np.sum(diff_orig_karaoke ** 2)
energy_diff_od = np.sum(diff_orig_diff ** 2)

corr_diff_ok = np.corrcoef(original, diff_orig_karaoke)[0, 1]
corr_diff_od = np.corrcoef(original, diff_orig_diff)[0, 1]


print("DIFFERENCE SIGNAL METRICS")

print(f"Difference 1 (Original - Karaoke / Vocals):")
print(f"   - Total Energy      : {energy_diff_ok:.4f}")
print(f"   - Correlation w/ Orig: {corr_diff_ok:.4f}")

print(f"Difference 2 (Original - Different Song):")
print(f"   - Total Energy      : {energy_diff_od:.4f}")
print(f"   - Correlation w/ Orig: {corr_diff_od:.4f}")


# Duration limit for clear rendering
duration = 5
samples_to_plot = min(sr * duration, length)
time = np.arange(samples_to_plot) / sr

# WINDOW 1: Original Song

plt.figure(figsize=(12, 4))
plt.plot(time, original[:samples_to_plot], color='red', linewidth=1.0)
plt.title("1. Original Song Signal")
plt.xlabel("Time (Seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# WINDOW 2: Karaoke Song

plt.figure(figsize=(12, 4))
plt.plot(time, karaoke[:samples_to_plot], color='gold', linewidth=1.0)
plt.title("2. Karaoke Song Signal")
plt.xlabel("Time (Seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# WINDOW 3: Different Song

plt.figure(figsize=(12, 4))
plt.plot(time, different[:samples_to_plot], color='blue', linewidth=1.0)
plt.title("3. Different Song Signal")
plt.xlabel("Time (Seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# WINDOW 4: Overlay (Original vs Karaoke) + Difference 1

plt.figure(figsize=(12, 6))

# Subplot A: Overlaid Signals
plt.subplot(2, 1, 1)
plt.plot(time, original[:samples_to_plot], color='red', linewidth=1.2, label="Original Song")
plt.plot(time, karaoke[:samples_to_plot], color='gold', linewidth=1.0, alpha=0.8, label="Karaoke Song")
plt.title("4A. Overlay: Original (Red) vs. Karaoke (Yellow)")
plt.xlabel("Time (Seconds)")
plt.ylabel("Amplitude")
plt.legend(loc="upper right")
plt.grid(True)

# Subplot B: Difference 1
plt.subplot(2, 1, 2)
plt.plot(time, diff_orig_karaoke[:samples_to_plot], color='magenta', linewidth=1.0)
plt.title("4B. Difference 1: Original - Karaoke (Extracted Vocals)")
plt.xlabel("Time (Seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()

# WINDOW 5: Overlay (Original vs Different Song) + Difference 2

plt.figure(figsize=(12, 6))

# Subplot A: Overlaid Signals
plt.subplot(2, 1, 1)
plt.plot(time, original[:samples_to_plot], color='red', linewidth=1.2, label="Original Song")
plt.plot(time, different[:samples_to_plot], color='blue', linewidth=1.0, alpha=0.7, label="Different Song")
plt.title("5A. Overlay: Original (Red) vs. Different Song (Blue)")
plt.xlabel("Time (Seconds)")
plt.ylabel("Amplitude")
plt.legend(loc="upper right")
plt.grid(True)

# Subplot B: Difference 2
plt.subplot(2, 1, 2)
plt.plot(time, diff_orig_diff[:samples_to_plot], color='orange', linewidth=1.0)
plt.title("5B. Difference 2: Original - Different Song")
plt.xlabel("Time (Seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()
