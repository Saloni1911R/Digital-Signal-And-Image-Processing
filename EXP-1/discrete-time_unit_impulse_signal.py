import numpy as np 
import matplotlib.pyplot as plt 

def unit_impulse(x_array): 
    signal = np.zeros_like(x_array) 
    zero_indices = np.where(x_array == 0)[0]
    
    if len(zero_indices) > 0:
        signal[zero_indices[0]] = 1
    else:
        print("Warning: n=0 is not within the specified start and stop range.")
        
    return signal 

start = -10  # Start value of the x-axis range 
stop = 10    # Stop value of the x-axis range 
step = 1     # Step size 

# Generate x-axis values 
x = np.arange(start, stop + step, step) 

# Generate unit impulse signal safely
impulse_signal = unit_impulse(x) 

# Plot
plt.stem(x, impulse_signal) 
plt.xlabel('Time (n)') 
plt.ylabel('Amplitude') 
plt.title('Unit Impulse Signal $\delta[n]$') 
plt.grid(True) 
plt.show()