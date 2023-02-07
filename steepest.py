import numpy as np
import interpolate as I
import pinpoint as P
import bracketing as B
from matplotlib import pyplot as plt
import numdifftools as nd
import funs as f

def steepest(x_current,tau,alpha_past,phi0,phip0,mu1,mu2,sigma):
    k = 0
    normf = 1
    x_past = x_current
    ggg = x_past
    while normf > tau:
        p_current = - f.fp(x_current) / np.linalg.norm(f.fp(x_current))
        p_past = - f.fp(x_past) / np.linalg.norm(f.fp(x_past))
        alpha_current = alpha_estimate(x_current,x_past,alpha_past,p_current,p_past)
        alpha_p, g = B.bracketing(x_current,alpha_current,phi0,phip0,p_current,mu1,mu2,sigma)
        x_past = x_current
        x_current = x_past + alpha_p * p_current
        normf = np.linalg.norm(f.fp(x_current))
        ggg = np.append(ggg,x_current)
        k = k + 1
        print(k)
    ggg = np.reshape(ggg,(k+1,2))
    return x_current, ggg

def alpha_estimate(x_current,x_past,alpha_past,p_current,p_past):
    alpha_current = alpha_past * np.dot(f.fp(x_past),p_past) / np.dot(f.fp(x_current),p_current)
    return alpha_current

x_current = np.array([3,-1])
p0 = -f.fp(x_current) / np.linalg.norm(f.fp(x_current))
tau = 0.01
alpha_past = 0.1
phi0 = f.phi(x_current,0,p0)
phip0 = f.phip(x_current,0,p0)
mu1 = 0.1
mu2 = 0.9
sigma = 2


x_last, ggg = steepest(x_current,tau,alpha_past,phi0,phip0,mu1,mu2,sigma)
# ggg = np.array([[0,0],[0,0]])
# print(ggg)

def fun(x1,x2):
    fun = np.zeros([len(x1),len(x2)])
    for i in range(len(x1)):
        for j in range(len(x2)):
            fun[i,j] = f.f([x1[i],x2[j]])
    return fun
x1 = np.linspace(-10,10,100)
x2 = np.linspace(-10,10,100)
plt.figure()
plt.contour(x1,x2,np.transpose(fun(x1,x2)),50)
for i in range(0,len(ggg)-1):
    plt.plot(ggg[i,0],ggg[i,1],marker=".",markersize=5,color='red')
plt.show()