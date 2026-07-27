import numpy as np 
import matplotlib.pyplot as plt 

def simulate_continuous_ramp(time, slope): 
    ramp = np.zeros_like(time) 
    ramp[time >= 0] = slope * time[time >= 0] 
    return ramp 

def simulate_discrete_ramp(n_axis, slope): 
    ramp = np.zeros_like(n_axis) 
    ramp[n_axis >= 0] = slope * n_axis[n_axis >= 0] 
    return ramp 

# --- Continuous Ramp Setup ---
time = np.linspace(-5, 5, 1000) 
slope = 2 
continuous_ramp = simulate_continuous_ramp(time, slope) 

# --- Discrete Ramp Setup ---
n = np.arange(-10, 11) # Samples from -10 to +10
discrete_ramp = simulate_discrete_ramp(n, slope) 

# --- Plotting ---
plt.figure(figsize=(10, 6)) 

# Continuous Plot
plt.subplot(2, 1, 1) 
plt.plot(time, continuous_ramp, linewidth=2) 
plt.title('Continuous Ramp Signal') 
plt.xlabel('Time (t)') 
plt.ylabel('Amplitude') 
plt.grid(True) 

# Discrete Plot
plt.subplot(2, 1, 2) 
plt.stem(n, discrete_ramp) 
plt.title('Discrete Ramp Signal') 
plt.xlabel('Sample (n)') 
plt.ylabel('Amplitude') 
plt.grid(True) 

plt.tight_layout() 
plt.show()