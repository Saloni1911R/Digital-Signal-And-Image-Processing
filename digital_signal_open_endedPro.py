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

n = np.arange(0,6)

x = [1,2,3,3,2,1]

plt.stem(n, x)

plt.title("Signal 2")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)
plt.show()


# ================


n = np.arange(-2,3)

x = []

for i in n:
    if i <= -1:
        x.append(-2)
    elif -1 < i <= 1:
        x.append(2*i)
    else:
        x.append(2)

plt.stem(n, x)

plt.title("Signal 3")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)
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