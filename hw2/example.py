#this is a program to illustrate a hello world type of optimization
#Joseph Carter
#version 1.0

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def f(x):
    return (1-x[0])**2 + (1-x[1])**2 + 0.5*(2*x[1]-x[0]**2)**2

x0 = [6,9]

res = minimize(f,x0)

print(res)

n1 = 100
n2 = 99
x1 = np.linspace(-5,5,n1)
x2 = np.linspace(-5,5,n2)

fun = np.zeros([n1,n2])

for i in range(n1):
    for j in range(n2):
        fun[i,j] = f([x1[i],x2[j]])

plt.figure()
plt.contour(x1,x2,np.transpose(fun),100)
plt.show()