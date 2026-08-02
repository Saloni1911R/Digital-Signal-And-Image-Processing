import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-3, 7)

x = [2,3,2,3,2,3,2,3,2,3]

plt.stem(n, x)

plt.title("Signal 1")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)
plt.show()


# =================

n = np.arange(0, 7)
x = [1, 2, 3, 3, 2, 1, 0]

plt.step(n, x, where="post")

plt.title("Signal 2")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.ylim(0, 4)  
plt.grid(True)
plt.show()


# ================

n = [-2, -2, -1, 1, 2, 2]

x = [0, -2, -2, 2, 2, 0]

plt.plot(n, x, color="red", linewidth=2)

plt.title("Signal 3")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)
plt.axhline(0, color="black", linewidth=0.8)  # X-axis line
plt.axvline(0, color="black", linewidth=0.8)  # Y-axis line
plt.show()

# =========================

def u(n):
    return np.where(n >= 0, 1, 0)

n = np.arange(-2,12)

x = u(n) - u(n-3) - 5*u(n-7)

plt.stem(n, x)

plt.title("Signal 4")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)
plt.show()

# =========================


import numpy as np
import matplotlib.pyplot as plt

def delta(n):
    return np.where(n == 0, 1, 0)

n = np.arange(-4,5)

x = delta(n) + 3*delta(n-1) + 5*delta(n+1)

plt.stem(n, x)

plt.title("Signal 5")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)
plt.show()