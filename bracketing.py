import numpy as np
import interpolate as I
import pinpoint as P
from matplotlib import pyplot as plt
import numdifftools as nd

# def f(x):
#     b = 1.5
#     return x[0]**2 + x[1]**2 + b * x[0] * x[1]
# def f(x):
#     return (1-x[0])**2 + 100 * (x[1] - x[0]**2)**2
def f(x):
    return (1-x[0])**2 + (1-x[1])**2 + 0.5*(2*x[1]-x[0]**2)**2

def fp(x):
    b = 1.5
    grad1 = 2 * x[0] + b * x[1]
    grad2 = 2 * x[1] + b * x[0]
    grad = np.array([grad1,grad2])
    grad = nd.Gradient(f)(x)
    return grad

def phi(x,alpha,p):
    return f(x + alpha * p)

def phip(x,alpha,p):
    return np.dot(fp(x + alpha * p),p)

def bracketing(x,alphainit,phi0,phip0,p,mu1,mu2,sigma):
    alpha1 = 0
    alpha2 = alphainit
    gg = alpha1
    phi1 = phi0
    phip1 = phip0
    first = True
    while True:
        phi2 = phi(x,alpha2,p)
        phip2 = phip(x,alpha2,p)
        if (phi2 > phi0 + (mu1 * alpha2 * phip0)) or ((first != True) and (phi2 > phi1)):
            alpha_s, g = P.pinpoint(alpha1,alpha2,x,p,phi0,phip0,mu1,mu2,phi1)
            gg = np.append(gg,alpha_s)
            return alpha_s, gg
        phip2 = phip(x,alpha2,p)
        if abs(phip2) <= -mu2 * phip0:
            alpha_s = alpha2
            gg = np.append(gg,alpha_s)
            return alpha_s, gg
        elif phip2 >= 0:
            alpha_s, g = P.pinpoint(alpha2,alpha1,x,p,phi0,phip0,mu1,mu2,phi2)
            gg = np.append(gg,alpha_s)
            return alpha_s, gg
        else:
            alpha1 = alpha2
            alpha2 = sigma * alpha2
        first = False

# x = np.array([-5,5])
# p = -fp(x) / np.linalg.norm(fp(x))
# phi0 = phi(x,0,p)
# phip0 = phip(x,0,p)
# mu1 = 0.1
# mu2 = 0.2
# sigma = 2
# alphainit = 0.2

# alpha_p, g = bracketing(x,alphainit,phi0,phip0,p,mu1,mu2,sigma)

# new_point = phi(x,alpha_p,p)

# plt.figure()
# alpha = np.linspace(0,10,100)
# y = np.zeros([len(alpha),1])
# plt.plot(alpha_p,new_point,marker="o")
# for i in range(0,len(alpha)):
#     y[i] = phi(x,alpha[i],p)
# for i in range(0,len(g)):
#     plt.plot(g[i],phi(x,g[i],p),marker="o")
# plt.plot(alpha,y)
# plt.show()
