import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import butter, sosfilt

# --- 1. Load Audio File ---
filename = 'onedance.mp3'
y, sr = librosa.load(filename, sr=None)
print(f"Sample Rate: {sr} Hz, Duration: {len(y)/sr:.2f} seconds")

# --- Helper Functions using SOS (Second-Order Sections) for numerical stability ---
def butter_lowpass_filter(data, cutoff, fs, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    return sosfilt(sos, data)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', analog=False, output='sos')
    return sosfilt(sos, data)

def butter_bandstop_filter(data, lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='stop', analog=False, output='sos')
    return sosfilt(sos, data)

# ==========================================
# PART (a): Low-Pass Filtering Experiments
# ==========================================
cutoff_frequencies = [500, 2000, 5000]  # Hz
lpf_outputs = {}

for fc in cutoff_frequencies:
    lpf_outputs[fc] = butter_lowpass_filter(y, cutoff=fc, fs=sr, order=6)
    sf.write(f'lpf_output_{fc}Hz.wav', lpf_outputs[fc], sr)

# Plot Spectrogram Comparison for LPF
plt.figure(figsize=(14, 8))
for idx, fc in enumerate(cutoff_frequencies, 1):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(lpf_outputs[fc])), ref=np.max)
    plt.subplot(3, 1, idx)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Low-Pass Filtered Spectrogram (Cutoff = {fc} Hz)')
    plt.ylim(0, 10000)
plt.tight_layout()
plt.show()

# ==========================================
# PART (b): Vocal & Background Suppression
# ==========================================
# Vocal range usually concentrated around 300 Hz – 3400 Hz
low_vocal, high_vocal = 300, 3400

# 1. Band-Pass Filter to extract/isolate Vocals
vocals_extracted = butter_bandpass_filter(y, low_vocal, high_vocal, fs=sr, order=6)

# 2. Band-Stop Filter to suppress Vocals (leaving Background Music)
background_isolated = butter_bandstop_filter(y, low_vocal, high_vocal, fs=sr, order=6)

# Save Outputs
sf.write('suppressed_background_only_vocals.wav', vocals_extracted, sr)
sf.write('suppressed_vocals_only_background.wav', background_isolated, sr)

# Plot Isolation Results
def plot_spec(signal, title):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_spec(vocals_extracted, 'Isolated/Extracted Vocal Band (300 Hz – 3400 Hz)')
plot_spec(background_isolated, 'Suppressed Vocals (Band-Stop: 300 Hz – 3400 Hz)')
