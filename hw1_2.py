#the purpose of this file is to solve the Brachistrone problem per ME575 hw1 problem 2.
#Joseph Carter
#Version 1.0

#import needed libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#given information
mu_k = 0
start = [0,1]
end = [1,0]
n = 12

x0 = [0.5,0.5]

h = start[1]
x = np.linspace(start[0],end[0],n)
y = np.linspace(start[1],end[1],n)



def f(x,y,h,mu_k):
    f = 0
    for i in range(0,n-1):
        deltax = x[i+1] - x[i]
        deltay = y[i+1] - y[i]
        f += (np.sqrt(deltax**2 + deltay**2) / (np.sqrt(h - y[i+1] - mu_k * x[i+1]) + np.sqrt(h - y[i] - mu_k * x[i])))
    return f

fun = np.zeros([len(x),len(y)])
for i in range(0,len(x)):
    for j in range(0,len(y)):
        fun[i,j] = f(x,y,h,mu_k)

print(fun)
plt.figure()
plt.contour(x,y,np.transpose(fun),100)
plt.show()