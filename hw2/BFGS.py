import numpy as np
import interpolate as I
import pinpoint as P
import bracketing as B
from matplotlib import pyplot as plt
import numdifftools as nd
import funs as f

def BFGS(tau,x,phi0,phip0,mu1,mu2):
    k = 0
    x0 = x.copy()
    ggg = x.copy()
    alpha_p = 1
    normf = 1
    reset = False
    while normf > tau:
        if k == 0 or reset == True:
            V0 = np.dot((1 / np.linalg.norm(f.fp(x))),np.eye(len(f.fp(x))))
            V = V0
            sigma = 2
        else:
            s = x - x0
            y = f.fp(x) - f.fp(x0)
            sigma = 1 / (np.dot(np.transpose(s),y))
            V = (np.eye(len(f.fp(x))) - sigma * np.matmul(s,np.transpose(y))) * V0 * (np.eye(len(f.fp(x))) - sigma * y * np.transpose(s)) + sigma * s * np.transpose(s)
        p = -V @ f.fp(x)
        if abs(f.fp(x) @ p) > 1E-6:
            reset = True
        alpha_p, g = B.bracketing(x,alpha_p,phi0,phip0,p,mu1,mu2,sigma)
        x0 = x
        x = x0 + alpha_p * p
        ggg = np.append(ggg,x)
        normf = np.linalg.norm(f.fp(x))
        k = k + 1
        print(k)
        if abs(f.f(x0) - f.f(x)) < 1E-3:
            break
    ggg = np.reshape(ggg,(k+1,2))
    return x,ggg

x = np.array([-0.5,-2.5])
p0 = -f.fp(x) / np.linalg.norm(f.fp(x))
tau = 1E-6
phi0 = f.phi(x,0,p0)
phip0 = f.phip(x,0,p0)
mu1 = 0.0001
mu2 = 0.001


x_last, ggg = BFGS(tau,x,phi0,phip0,mu1,mu2)

def fun(x1,x2):
    fun = np.zeros([len(x1),len(x2)])
    for i in range(len(x1)):
        for j in range(len(x2)):
            fun[i,j] = f.f([x1[i],x2[j]])
    return fun
x1 = np.linspace(-2,2.5,100)
x2 = np.linspace(-1,3,100)
plt.figure()
plt.contour(x1,x2,np.transpose(fun(x1,x2)),50)
for i in range(0,len(ggg)-1):
    plt.plot(ggg[i,0],ggg[i,1],color='red',markersize=5,marker=".")
plt.show()
