#the purpose of this file is to perform line search for hw2 problem 1
#Joseph Carter
#Version 1.0

#import needed libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

def f(x1,x2):
    return x1**2 + x2**2

# def fun(x1,x2):
#     fun = np.zeros([len(x1),len(x2)])
#     for i in range(len(x1)):
#         for j in range(len(x2)):
#             fun[i,j] = f(x1[i],x2[j])
#     return fun
# x1 = np.linspace(-10,10,100)
# x2 = np.linspace(-10,10,100)
# plt.figure()
# plt.contour(x1,x2,np.transpose(fun(x1,x2)),50)
# plt.show()


