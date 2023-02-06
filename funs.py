import numpy as np
import interpolate as I
from matplotlib import pyplot as plt
import numdifftools as nd

# def f(x):
#     b = 1.2
#     return x[0]**2 + x[1]**2 + b * x[0] * x[1]
# def f(x):
#     return (1-x[0])**2 + 100 * (x[1] - x[0]**2)**2
def f(x):
    return (1-x[0])**2 + (1-x[1])**2 + 0.5*(2*x[1]-x[0]**2)**2


# def fp(x):
#     b = 1.2
#     grad1 = 2 * x[0] + b * x[1]
#     grad2 = 2 * x[1] + b * x[0]
#     grad = np.array([grad1,grad2])
#     # grad = nd.Gradient(f)(x)
#     return grad
# def fp(x):
#     grad1 = -2*(1-x[0]) + 200*(x[1] - x[0]**2) * -2*x[0]
#     grad2 = 200 * (x[1] - x[0]**2)
#     grad = np.array([grad1,grad2])
#     return grad
def fp(x):
    grad1 = -2*(1-x[0]) + (2*x[1]-x[0]**2)*(-2*x[0])
    grad2 = -2*(1-x[1]) + (2*x[1]-x[0]**2)*(2)
    grad = np.array([grad1,grad2])
    return grad

def phi(x,alpha,p):
    return f(x + alpha * p)

def phip(x,alpha,p):
    return np.dot(fp(x + alpha * p),p)