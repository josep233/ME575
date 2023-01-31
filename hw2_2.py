import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

x1 = np.linspace(-10,10,100)
x2 = np.linspace(-10,10,100)
f = np.zeros([len(x1),len(x2)])
for i in range(0,len(x1)):
    for j in range(0,len(x2)):
        f[i,j] = x1[i]**2 + x2[j]**2 + 1.5 * x1[i] * x2[j]

alpha0 = 0.5
x0 = np.array([5,-5])

def phi(alpha):
    return f(x0 + alpha * pk)

plt.figure()
plt.contour(x1,x2,f,50)
plt.plot(x0[0],x0[1],marker="o")
plt.show()

