import numpy as np 
import matplotlib.pyplot as plt 

def simulate_continuous_sine_wave(time, amplitude, frequency, phase): 
    return amplitude * np.sin(2 * np.pi * frequency * time + phase) 

def simulate_discrete_sine_wave(n_axis, sampling_frequency, amplitude, frequency, phase): 
    # Calculate true time matching the sample grid index
    time_samples = n_axis / sampling_frequency 
    return amplitude * np.sin(2 * np.pi * frequency * time_samples + phase) 

# Parameters
amplitude = 1 
frequency = 2        # 2 Hz signal
phase = 0 
sampling_frequency = 20  # 20 Hz (avoids blocky layout/aliasing)

# Set up synchronized axes spanning exactly 0 to 1 second
time_continuous = np.linspace(0, 1, 1000) 
n_discrete = np.arange(0, int(1 * sampling_frequency) + 1) 

# Simulate both signals
continuous_sine_wave = simulate_continuous_sine_wave(time_continuous, amplitude, frequency, phase) 
discrete_sine_wave = simulate_discrete_sine_wave(n_discrete, sampling_frequency, amplitude, frequency, phase) 

# Plotting
plt.figure(figsize=(10, 6)) 

plt.subplot(2, 1, 1) 
plt.plot(time_continuous, continuous_sine_wave, 'b') 
plt.title('Continuous Sine Wave Signal') 
plt.xlabel('Time (s)') 
plt.ylabel('Amplitude') 
plt.grid(True)

plt.subplot(2, 1, 2) 
plt.stem(n_discrete / sampling_frequency, discrete_sine_wave, 'r') 
plt.title('Discrete Sine Wave Signal') 
plt.xlabel('Time (s)') 
plt.ylabel('Amplitude') 
plt.grid(True)

plt.tight_layout() 
plt.show()