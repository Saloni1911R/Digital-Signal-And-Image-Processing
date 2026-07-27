import numpy as np 
import matplotlib.pyplot as plt 

def simulate_impulse_train(time_axis, period): 
    impulse_train = np.zeros_like(time_axis) 
    for k in range(int(time_axis.min() // period), int(time_axis.max() // period) + 1):
        target_time = k * period
        idx = np.where(np.isclose(time_axis, target_time))[0]
        if len(idx) > 0:
            impulse_train[idx[0]] = 1
            
    return impulse_train 
start = -50
stop = 50
period = 10 
step = 1

# Generate explicit time axis values (n)
n = np.arange(start, stop + step, step) 

# Simulate the impulse train 
impulse_train = simulate_impulse_train(n, period) 

# Plot and display the impulse train
plt.stem(n, impulse_train) 
plt.title(r'Discrete Time Impulse Train $\sum_{k=-\infty}^{\infty} \delta[n - kP]$') 
plt.xlabel('Sample (n)') 
plt.ylabel('Amplitude') 
plt.grid(True)
plt.show()