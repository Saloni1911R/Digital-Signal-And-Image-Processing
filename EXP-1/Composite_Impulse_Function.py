import numpy as np 
import matplotlib.pyplot as plt 

def simulate_impulse_function(n_axis): 
    y = np.zeros_like(n_axis) 
    y[n_axis == 0] = 1 
    y[n_axis == 1] += 1 
    y[n_axis == -5] += 3 
    return y 

# Define safe discrete sample time coordinates
n = np.arange(-10, 11) 

# Simulate the function 
function_values = simulate_impulse_function(n) 

# Plot and display the function using stem components
plt.figure(figsize=(10, 4))
plt.stem(n, function_values, 'm') 
plt.title('Function $y[n] = \delta[n] + \delta[n-1] + 3\delta[n+5]$') 
plt.xlabel('Sample Index (n)') 
plt.ylabel('Amplitude') 
plt.ylim([-0.5, 4.5]) 
plt.grid(True) 
plt.show()