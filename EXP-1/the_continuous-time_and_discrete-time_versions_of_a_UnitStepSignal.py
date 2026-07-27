import numpy as np 
import matplotlib.pyplot as plt 

def simulate_continuous_unit_step(time): 
    unit_step = np.zeros_like(time) 
    unit_step[time >= 0] = 1 
    return unit_step 

def simulate_discrete_unit_step(n_axis): 
    # FIX: Base the step on actual time values, not array length!
    unit_step = np.zeros_like(n_axis) 
    unit_step[n_axis >= 0] = 1 
    return unit_step 

# --- Continuous Signal ---
# Time range from -5 to 5
time = np.linspace(-5, 5, 1000) 
continuous_unit_step = simulate_continuous_unit_step(time) 

# --- Discrete Signal ---
# FIX: Define an explicit sample axis spanning negative and positive values
n = np.arange(-10, 11)
discrete_unit_step = simulate_discrete_unit_step(n)

# --- Plotting ---
plt.figure(figsize=(10, 6)) 

# Continuous Plot
plt.subplot(2, 1, 1) 
plt.plot(time, continuous_unit_step, linewidth=2) 
plt.title('Continuous Unit Step Signal $u(t)$') 
plt.xlabel('Time (t)') 
plt.ylabel('Amplitude') 
plt.grid(True)

# Discrete Plot
plt.subplot(2, 1, 2) 
# FIX: Pass 'n' explicitly so the step aligns perfectly at 0 on the x-axis
plt.stem(n, discrete_unit_step) 
plt.title('Discrete Unit Step Signal $u[n]$') 
plt.xlabel('Sample (n)') 
plt.ylabel('Amplitude') 
plt.grid(True)

plt.tight_layout() 
plt.show()