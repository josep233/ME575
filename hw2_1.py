#the purpose of this file is to perform line search for hw2 problem 1
#Joseph Carter
#Version 1.0

#import needed libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

def f(x1,x2,beta):
    return x1**2 + x2**2 - beta*x1*x2

def fun(x1,x2,beta):
    fun = np.zeros([len(x1),len(x2)])
    for i in range(len(x1)):
        for j in range(len(x2)):
            fun[i,j] = f(x1[i],x2[j],beta)
    return fun


x1 = np.linspace(-10,10,100)
x2 = np.linspace(-10,10,100)
beta = 3/2
x0 = np.array([5,-5])
tau = 0.001
fun = fun(x1,x2,beta)


plt.figure()
plt.contour(x1,x2,np.transpose(fun),50)
plt.show()

def linesearch(pk,alpha0):
    alphak = 0
    return alphak

def steepest_descent(x0,tau):
    xs = 0
    return xs,fun(xs)

def pinpoint(alphalow,alphahigh,phi0,philow,phihigh,phip0,phiplow,phiphigh,mu1,mu2):
    alphas = 0
    return alphas

def bracketing(alpha0,phi0,phip0,mu1,mu2,sigma):
    alphas = 0
    return alphas

def backtracking(alpha0,mu1,ro):
    alphas = 0
    return alphas
