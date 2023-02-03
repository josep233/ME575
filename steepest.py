import numpy as np
import interpolate as I
import pinpoint as P
import bracketing as B
from matplotlib import pyplot as plt
import numdifftools as nd

def f(x):
    b = 1.5
    return x[0]**2 + x[1]**2 + b * x[0] * x[1]
# def f(x):
#     return (1-x[0])**2 + 100 * (x[1] - x[0]**2)**2
# def f(x):
#     return (1-x[0])**2 + (1-x[1])**2 + 0.5*(2*x[1]-x[0]**2)**2

def fp(x):
    # b = 1.5
    # grad1 = 2 * x[0] + b * x[1]
    # grad2 = 2 * x[1] + b * x[0]
    # grad = np.array([grad1,grad2])
    grad = nd.Gradient(f)(x)
    return grad

def phi(x,alpha,p):
    return f(x + alpha * p)

def phip(x,alpha,p):
    return np.dot(fp(x + alpha * p),p)

def steepest(x_current,tau,alpha_past,phi0,phip0,mu1,mu2,sigma):
    k = 0
    normf = 1
    x_past = x_current
    ggg = x_past
    while normf > tau:
        p_current = - fp(x_current) / np.linalg.norm(fp(x_current))
        p_past = - fp(x_past) / np.linalg.norm(fp(x_past))
        alpha_current = alpha_estimate(x_current,x_past,alpha_past,p_current,p_past)
        alpha_p, g = B.bracketing(x_current,alpha_current,phi0,phip0,p_current,mu1,mu2,sigma)
        x_past = x_current
        x_current = x_past + alpha_p * p_current
        normf = np.linalg.norm(fp(x_current))
        ggg = np.append(ggg,x_current)
        k = k + 1
        print(k)
    ggg = np.reshape(ggg,(k+1,2))
    return x_current, g, ggg

def alpha_estimate(x_current,x_past,alpha_past,p_current,p_past):
    alpha_current = alpha_past * np.dot(fp(x_past),p_past) / np.dot(fp(x_current),p_current)
    return alpha_current

x_current = np.array([-5.5,-2])
p0 = -fp(x_current) / np.linalg.norm(x_current)
tau = 1E-6
alpha_past = 0.1
phi0 = phi(x_current,0,p0)
phip0 = phip(x_current,0,p0)
mu1 = 0.1
mu2 = 0.2
sigma = 2


x_last, g, ggg = steepest(x_current,tau,alpha_past,phi0,phip0,mu1,mu2,sigma)
# ggg = np.array([[0,0],[0,0]])
# print(ggg)

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
for i in range(0,len(ggg)):
    plt.plot(ggg[i,0],ggg[i,1],marker="o")
plt.show()