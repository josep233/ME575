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
alpha0 = 0.5
grad0 = np.array([1,1])
x0 = 5
y0 = -5
tau = 0.01
fun = fun(x1,x2,beta)


plt.figure()
plt.contour(x1,x2,np.transpose(fun),50)
plt.show()

def linesearch(pk,alpha0):
    #dependent on steepest descent
    alphak = 0
    return alphak

def steepest_descent(x0,tau,alpha0,grad0):
    k = 0
    grad = np.array(np.gradient(fun)[0][x0,y0],np.gradient(fun)[1][x0,y0])
    while np.linalg.norm(grad) > tau:
        pk = - grad / np.linalg.norm(grad)
        p0 = - grad0 / np.linalg.norm(grad0)
        alpha = alpha0 * np.dot(grad,pk) / np.dot(grad0,p0)
        alpha = linesearch(pk,alpha)
        return xs,fun(xs)

def pinpoint(alphalow,alphahigh,phi0,philow,phihigh,phip0,phiplow,phiphigh,mu1,mu2):
    #dependent on interpolation
    alphas = 0
    return alphas

def bracketing(alpha0,phi0,phip0,mu1,mu2,sigma):
    alpha1 = 0
    alpha2 = alpha0
    phi1 = phi0
    phip1 = phip0
    first = True
    while first == True:
        phi2 = phi(alpha2)
    alphas = 0
    return alphas
