#the purpose of this file is to perform line search for hw2 problem 1
#Joseph Carter
#Version 1.0

#import needed libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time 

# alpha0 = 0.5
# x0 = np.array([-2.5,5])
# p0 = np.array([0,1])
# mu1 = 0.1
# mu2 = 0.2
# sigma = 2
# tau = 1E-6
#==================================================================================================================================================================================================================================================================================
def f(x):
    b = 3/2
    return x[0]**2 + x[1]**2 + b * x[0] * x[1]
# def f(x):
#     return (1-x[0])**2 + 100 * (x[1] - x[0]**2)**2
# def f(x):
#     return (1-x[0])**2 + (1-x[1])**2 + 0.5 * (2*x[1]-x[0]**2)**2
def fp(x):
    delta = 0.000001
    num_terms = len(x)
    xs1 = x.copy()
    xs2 = x.copy()
    comp = np.zeros([num_terms])
    for i in range(0,num_terms):
        xs1[i] = xs1[i] + delta
        xs2[i] = xs2[i] - delta
        comp[i] = (f(xs1) - f(xs2)) / (2*delta)
        xs1[i] = xs1[i] - delta
        xs2[i] = xs2[i] + delta
    return comp
#==================================================================================================================================================================================================================================================================================
def phi(x,alpha,p):
    return f(x + alpha * p)
def phip(x,alpha,p):
    return np.dot(fp(x + alpha * p),p)
#==================================================================================================================================================================================================================================================================================
def beta1(phiplow,phiphigh,philow,phihigh,alphalow,alphahigh):
    return phiplow + phiphigh - 3 * ((philow - phihigh)/(alphalow - alphahigh))
def beta2(phiplow,phiphigh,philow,phihigh,alphalow,alphahigh):
    return np.sign(alphahigh - alphalow) * np.sqrt(beta1(phiplow,phiphigh,philow,phihigh,alphalow,alphahigh)**2 - phiplow*phiphigh)
def interp(phiplow,phiphigh,philow,phihigh,alphalow,alphahigh):
    return alphahigh - (alphahigh - alphalow) * ((phiphigh + beta2(phiplow,phiphigh,philow,phihigh,alphalow,alphahigh) - beta1(phiplow,phiphigh,philow,phihigh,alphalow,alphahigh))/(phiphigh - phiplow + 2*beta2(phiplow,phiphigh,philow,phihigh,alphalow,alphahigh)))
#==================================================================================================================================================================================================================================================================================
def pinpoint(x0,p,alphalow,alphahigh,mu1,mu2,phi0,phip0):
    k = 0
    while True:
        philow = phi(x0,alphalow,p)
        phiplow = phip(x0,alphalow,p)
        phihigh = phi(x0,alphahigh,p)
        phiphigh = phi(x0,alphahigh,p)

        alphap = interp(phiplow,phiphigh,philow,phihigh,alphalow,alphahigh)
        phi_p = phi(x0,alphap,p)

        if (phi_p > (phi0 + mu1 * alphap * phip0)) or (phi_p > philow):
            alphahigh = alphap
        else:
            phip_p = phip(x0,alphap,p)
            if np.abs(phip_p) <= -1 * mu2 * phip0:
                alphas = alphap
                return alphas
            elif phip_p * (alphahigh - alphalow) >= 0:
                alphahigh = alphalow
            alphalow = alphap
        k = k + 1
#==================================================================================================================================================================================================================================================================================
def bracketing(alpha0,phi0,phip0,mu1,mu2,sigma,p):
    alpha1 = 0
    alpha2 = alpha0
    phi1 = phi0
    phip1 = phip0
    first = True
    while True:
        phi2 = phi(x0,alpha2,p)
        if (phi2 > phi0 + mu1*alpha2*phip0) or ((first == False) and (phi2 > phi1)):
            alphas = pinpoint(x0,p,alpha1,alpha2,mu1,mu2,phi0,phip0)
            return alphas
        phip2 = phip(x0,alpha2,p)
        if abs(phip2) <= -mu2 * phip0:
            return alpha2
        elif phip2 >= 0:
            alphas = pinpoint(x0,p,alpha2,alpha1,mu1,mu2,phi0,phip0)
            return alphas
        else:
            alpha1 = alpha2
            alpha2 = sigma * alpha2
        first = False
#==================================================================================================================================================================================================================================================================================
def steepest(tau):
    k = 0
    x = x0
    xk = x0
    fk = f(xk)
    nfp = 1
    while nfp > tau:
        p = -fp(x0) / np.linalg.norm(fp(x0))
        alpha0 = alpha_estimate(p,x)
        alpha = bracketing(alpha0,phi0,phip0,mu1,mu2,sigma,p)
        x = x + alpha * p
        xk = np.stack((xk,x),axis=0)
        fk = np.append(fk,f(x))
        k = k + 1
        nfp = np.linalg.norm(fp(x))
        return xk, fk

def alpha_estimate(p,x):
    alpha = alpha0 * np.dot(fp(x0),p0) / np.dot(fp(x),p)
    return alpha
#==================================================================================================================================================================================================================================================================================
alpha0 = 0.3
x0 = np.array([-2.5,5])
p = -fp(x0) / np.linalg.norm(fp(x0))
p0 = np.array([0,-1])
mu1 = 0.1
mu2 = 0.2
sigma = 2
phi0 = phi(x0,0,p0)
phip0 = phip(x0,0,p0)
tau = 1E-6
alpha = bracketing(alpha0,phi0,phip0,mu1,mu2,sigma,p0)
xnew = x0 + alpha * p0

def fun(x1,x2):
    fun = np.zeros([len(x1),len(x2)])
    for i in range(len(x1)):
        for j in range(len(x2)):
            fun[i,j] = f([x1[i],x2[j]])
    return fun
x1 = np.linspace(-10,10,100)
x2 = np.linspace(-10,10,100)
plt.figure()
plt.contour(x1,x2,np.transpose(fun(x1,x2)),50)
plt.plot(x0[0],x0[1],marker="o")
plt.plot(p0*x0)
plt.show()
