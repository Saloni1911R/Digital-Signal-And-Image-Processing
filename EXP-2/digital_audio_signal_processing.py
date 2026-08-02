import wave
import numpy as np
import matplotlib.pyplot as plt

audio_path = "/home/saloni/DSIP/EXP-2/music.wav" 

with wave.open(audio_path, 'rb') as wav_file:
    n_channels = wav_file.getnchannels()
    sample_width = wav_file.getsampwidth()
    frame_rate = wav_file.getframerate()
    n_frames = wav_file.getnframes()
    
    raw_data = wav_file.readframes(n_frames)
    
    samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
    
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

samples_norm = samples / np.max(np.abs(samples))

np.random.seed(42)

num_kernels = 15
kernel_length = 10
kernels = {}

for k in range(1, num_kernels + 1):
    raw_kernel = np.random.randint(0, 256, size=kernel_length).astype(np.float32)
    kernel = raw_kernel / np.sum(raw_kernel)
    kernels[f"Impulse_Response_{k}"] = kernel


plt.figure(figsize=(10, 4))
plt.plot(samples_norm, color='black', alpha=0.8)
plt.title("Original Audio Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show() 

for name, kernel in kernels.items():
    # Perform 1D Convolution using NumPy
    convoluted = np.convolve(samples_norm, kernel, mode='same')
    
    max_val = np.max(np.abs(convoluted))
    convoluted_norm = convoluted / max_val if max_val > 0 else convoluted
    
    convoluted_int16 = (convoluted_norm * 32767).astype(np.int16)
    out_file = f"output_{name}.wav"
    
    with wave.open(out_file, 'wb') as out_wav:
        out_wav.setnchannels(1)
        out_wav.setsampwidth(2)  
        out_wav.setframerate(frame_rate)
        out_wav.writeframes(convoluted_int16.tobytes())
        
    print(f"Saved: {out_file}")
    
    plt.figure(figsize=(10, 4))
    plt.plot(convoluted_norm, color='blue', linewidth=0.8)
    plt.title(f"Convolved Signal: {name}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()  