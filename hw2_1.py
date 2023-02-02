#the purpose of this file is to perform line search for hw2 problem 1
#Joseph Carter
#Version 1.0

#import needed libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numdifftools as nd

#==================================================================================================================================================================================================================================================================================
def f(x):
    b = 0
    return x[0]**2 + x[1]**2 + b * x[0] * x[1]
# def f(x):
#     return (1-x[0])**2 + 100 * (x[1] - x[0]**2)**2
# def f(x):
#     return (1-x[0])**2 + (1-x[1])**2 + 0.5 * (2*x[1]-x[0]**2)**2
def fp(x):
    grad = nd.Gradient(fun=f)(x)
    return grad
#==================================================================================================================================================================================================================================================================================
def phi(x,alpha,p):
    return f(x + alpha * p)
def phip(x,alpha,p):
    
    return np.dot(fp(x + alpha * p),p)
#==================================================================================================================================================================================================================================================================================
def interp(phiplow,phiphigh,philow,phihigh,alphalow,alphahigh):
    beta1 = phiplow + phiphigh - 3 * ((philow - phihigh)/(alphalow - alphahigh))
    beta2 = np.sign(alphahigh - alphalow) * np.sqrt(beta1**2 - phiplow*phiphigh)
    alphap = alphahigh - (alphahigh - alphalow) * ((phiphigh + beta2 - beta1)/(phiphigh - phiplow + 2*beta2))
    return alphap
#==================================================================================================================================================================================================================================================================================
def pinpoint(x0,p,alphalow,alphahigh,mu1,mu2,phi0,phip0,philow,phiplow,phihigh,phiphigh):
    k = 0
    j = 0
    while True:
        alphap = interp(phiplow,phiphigh,philow,phihigh,alphalow,alphahigh)
        phi_p = phi(x0,alphap,p)
        if (phi_p > (phi0 + mu1 * alphap * phip0)) or (phi_p > philow):
            j = j + 1
            if j > 3:
                print(alphap)
                return alphap
            alphahigh = alphap
            phihigh = phi_p
        else:
            phip_p = phip(x0,alphap,p)
            if np.abs(phip_p) <= -1 * mu2 * phip0:
                return alphap
            elif phip_p * (alphahigh - alphalow) >= 0:
                alphahigh = alphalow
            alphalow = alphap
        k = k + 1


#==================================================================================================================================================================================================================================================================================
def bracketing(alphainit,phi0,phip0,mu1,mu2,sigma,p):
    alphalow = 0
    alphahigh = alphainit
    philow = phi0
    phiplow = phip0
    first = True
    while True:
        phihigh = phi(x0,alphahigh,p)
        phiphigh = phip(x0,alphahigh,p)
        if (phihigh > phi0 + mu1*alphahigh*phip0) or ((first == False) and (phihigh > philow)):
            alphas = pinpoint(x0,p,alphalow,alphahigh,mu1,mu2,phi0,phip0,philow,phiplow,phihigh,phiphigh)
            return alphas
        phiphigh = phip(x0,alphahigh,p)
        if abs(phiphigh) <= -mu2 * phip0:
            return alphahigh
        elif phiphigh >= 0:
            alphas = pinpoint(x0,p,alphahigh,alphalow,mu1,mu2,phi0,phip0,philow,phiplow,phihigh,phiphigh)
            return alphas
        else:
            alphalow = alphahigh
            alphahigh = sigma * alphahigh
        first = False
#==================================================================================================================================================================================================================================================================================
def steepest(tau,x0,alpha0,mu1,mu2,sigma):
    k = 0
    x = x0.copy()
    xk = x0.copy()
    fk = f(xk)
    nfp = 1
    alpha = alpha0
    while nfp > tau:
        phi0 = phi(x0,0,-fp(x0) / np.linalg.norm(fp(x0)))
        phip0 = phip(x0,0,-fp(x0) / np.linalg.norm(fp(x0)))
        p = -fp(x) / np.linalg.norm(fp(x))
        p0 = -fp(x0) / np.linalg.norm(fp(x0))
        alphainit = alpha_estimate(p,x,x0,p0,alpha)
        alpha = bracketing(alphainit,phi0,phip0,mu1,mu2,sigma,p)
        x0 = x
        x = x0 + alpha * p
        k = k + 1
        nfp = np.linalg.norm(fp(x))
    return xk, fk

def alpha_estimate(p,x,x0,p0,alpha0):
    alpha = alpha0 * (np.dot(fp(x0),p0) / np.dot(fp(x),p))
    return alpha

alpha0 = 0.1
# x0 = np.array([-0.80537252,0.6057])
x0 = np.array([-5,5])
mu1 = 0.1
mu2 = 0.9
sigma = 2
tau = 1E-6

xk,fk = steepest(tau,x0,alpha0,mu1,mu2,sigma)
print(fk)
#==================================================================================================================================================================================================================================================================================
# alpha0 = 0.3
# x0 = np.array([-7.5,2.5])
# mu1 = 0.1
# mu2 = 0.2
# sigma = 2
# tau = 1E-6

# xk,fk = steepest(tau,x0)

# print(xk,fk)

def fun(x1,x2):
    fun = np.zeros([len(x1),len(x2)])
    for i in range(len(x1)):
        for j in range(len(x2)):
            fun[i,j] = f([x1[j],x2[i]])
    return fun
x1 = np.linspace(-10,10,100)
x2 = np.linspace(-10,10,100)
plt.figure()
plt.contour(x1,x2,np.transpose(fun(x1,x2)),50)
plt.plot(x0[0],x0[1],marker="o")
# plt.plot(xk[1][0],xk[1][1],marker="o")
plt.show()
