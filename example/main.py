import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

x = np.array(range(-10,10))
print(x)
y = x**2

plt.plot(x,y, color = "Green", label = "quadratic func")
plt.xlabel("x")
plt.ylabel("y")

plt.legend()
plt.grid(True)

plt.show()