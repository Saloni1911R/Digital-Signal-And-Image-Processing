from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# 1. Load Audio File
audio_path = "/home/saloni/DSIP/EXP-2/music.mp3"  # Replace with your audio file path
audio = AudioSegment.from_mp3(audio_path).set_channels(1)
samples = np.array(audio.get_array_of_samples()).astype(np.float32)

# Normalize original signal
samples_norm = samples / np.max(np.abs(samples))

# 2. Define Multiple Impulse Responses (Kernels)
kernels = {
    "Comb_Filter": np.array([1, 0, 1, 0, 1], dtype=np.float32),
    "High_Pass": np.array([1, -1], dtype=np.float32),
    "Low_Pass": np.ones(5, dtype=np.float32) / 5.0,
    "Notch_Like": np.array([-0.5, 1, -0.5], dtype=np.float32)
}

# 3. Apply Kernels One by One
results = {}

plt.figure(figsize=(12, 2 * (len(kernels) + 1)))
plt.subplot(len(kernels) + 1, 1, 1)
plt.plot(samples_norm, color='blue')
plt.title("Original Audio Signal")

for i, (name, kernel) in enumerate(kernels.items(), start=2):
    # Perform convolution
    convoluted = convolve(samples, kernel, mode='same')
    
    # Normalize
    if np.max(np.abs(convoluted)) > 0:
        convoluted_norm = convoluted / np.max(np.abs(convoluted))
    else:
        convoluted_norm = convoluted
        
    results[name] = convoluted_norm
    
    # Export as WAV
    convoluted_int16 = (convoluted_norm * 32767).astype(np.int16)
    out_audio = AudioSegment(
        convoluted_int16.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=2,
        channels=1
    )
    out_file = f"output_{name}.wav"
    out_audio.export(out_file, format="wav")
    print(f"Saved: {out_file}")
    
    # Plot waveform
    plt.subplot(len(kernels) + 1, 1, i)
    plt.plot(convoluted_norm)
    plt.title(f"Filtered: {name}")

plt.tight_layout()
plt.show()