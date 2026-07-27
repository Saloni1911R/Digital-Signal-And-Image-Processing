import numpy as np 
import matplotlib.pyplot as plt

def simulate_continuous_parabolic(time, coefficients): 
    return np.polyval(coefficients, time) 

def simulate_discrete_parabolic(n_axis, coefficients):
    return np.polyval(coefficients, n_axis) 

# --- Continuous Setup ---
time = np.linspace(-5, 5, 1000) # Time range from -5 to 5 
coefficients = [1, 2, 1]        # Represents: 1*t^2 + 2*t + 1
continuous_parabolic = simulate_continuous_parabolic(time, coefficients) 

# --- Discrete Setup ---
n = np.arange(-5, 6, 1) # Samples at -5, -4, -3... up to 5
discrete_parabolic = simulate_discrete_parabolic(n, coefficients) 

# --- Plotting ---
plt.figure(figsize=(10, 6)) 

# Continuous Plot
plt.subplot(2, 1, 1) 
plt.plot(time, continuous_parabolic, linewidth=2) 
plt.title('Continuous Parabolic Signal ($t^2 + 2t + 1$)') 
plt.xlabel('Time (t)') 
plt.ylabel('Amplitude') 
plt.grid(True) 

# Discrete Plot
plt.subplot(2, 1, 2) 
plt.stem(n, discrete_parabolic) 
plt.title('Discrete Parabolic Signal ($n^2 + 2n + 1$)') 
plt.xlabel('Sample (n)') 
plt.ylabel('Amplitude') 
plt.grid(True) 

plt.tight_layout() 
plt.show()