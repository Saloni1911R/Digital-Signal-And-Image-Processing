import numpy as np 
import matplotlib.pyplot as plt 

def simulate_continuous_exponential(time, amplitude, coefficient): 
    return amplitude * np.exp(coefficient * time) 

def simulate_discrete_exponential(n_axis, amplitude, coefficient): 
    return amplitude * np.exp(coefficient * n_axis) 

# --- Continuous Setup ---
time = np.linspace(0, 5, 1000) 
amplitude = 2 
coefficient = -0.5 
continuous_exponential = simulate_continuous_exponential(time, amplitude, coefficient) 

# --- Discrete Setup ---
# Let's take 4 samples per second (21 samples total from 0 to 5)
n = np.linspace(0, 5, 21) 
discrete_exponential = simulate_discrete_exponential(n, amplitude, coefficient) 

# --- Plotting ---
plt.figure(figsize=(10, 6)) 

# Continuous Plot
plt.subplot(2, 1, 1) 
plt.plot(time, continuous_exponential, linewidth=2) 
plt.title('Continuous Exponential Signal $x(t) = A \cdot e^{bt}$') 
plt.xlabel('Time (t)') 
plt.ylabel('Amplitude') 
plt.grid(True) 

# Discrete Plot
plt.subplot(2, 1, 2) 
plt.stem(n, discrete_exponential) 
plt.title('Discrete Exponential Signal $x[n] = A \cdot e^{bn}$') 
plt.xlabel('Time Sample (n)') 
plt.ylabel('Amplitude') 
plt.grid(True) 

plt.tight_layout() 
plt.show()