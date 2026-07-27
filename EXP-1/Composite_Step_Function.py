import numpy as np 
import matplotlib.pyplot as plt 

def simulate_step_function(time_axis): 
    y = np.zeros_like(time_axis) 
    y[time_axis >= -5] += 3 
    y[time_axis >= 0] += 1 
    y[time_axis >= 1] += 1 
    return y 

# Define a stable time window from -10 to 10
time = np.linspace(-10, 10, 1000) 

# Simulate the function 
function_values = simulate_step_function(time) 

# Plot and display the function 
plt.figure(figsize=(10, 4))
plt.plot(time, function_values, 'g-', linewidth=2) 
plt.title('Function $y(t) = u(t) + u(t-1) + 3u(t+5)$') 
plt.xlabel('Time (t)') 
plt.ylabel('Amplitude') 
plt.ylim([-0.5, 5.5]) 
plt.grid(True) 
plt.show()