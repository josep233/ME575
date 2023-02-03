import numpy as np
import interpolate as I
import pinpoint as P
from matplotlib import pyplot as plt
import numdifftools as nd
import funs as f

def bracketing(x,alphainit,phi0,phip0,p,mu1,mu2,sigma):
    alpha1 = 0
    alpha2 = alphainit
    gg = alpha1
    phi1 = phi0
    phip1 = phip0
    first = True
    while True:
        phi2 = f.phi(x,alpha2,p)
        phip2 = f.phip(x,alpha2,p)
        if (phi2 > phi0 + (mu1 * alpha2 * phip0)) or ((first != True) and (phi2 > phi1)):
            alpha_s, g = P.pinpoint(alpha1,alpha2,x,p,phi0,phip0,mu1,mu2,phi1)
            gg = np.append(gg,alpha_s)
            return alpha_s, gg
        phip2 = f.phip(x,alpha2,p)
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
# p = -f.fp(x) / np.linalg.norm(f.fp(x))
# phi0 = f.phi(x,0,p)
# phip0 = f.phip(x,0,p)
# mu1 = 0.1
# mu2 = 0.9
# sigma = 2
# alphainit = 10

# alpha_p, g = bracketing(x,alphainit,phi0,phip0,p,mu1,mu2,sigma)

# new_point = f.phi(x,alpha_p,p)

# plt.figure()
# alpha = np.linspace(0,10,100)
# y = np.zeros([len(alpha),1])
# plt.plot(alpha_p,new_point,marker="o")
# for i in range(0,len(alpha)):
#     y[i] = f.phi(x,alpha[i],p)
# for i in range(0,len(g)):
#     plt.plot(g[i],f.phi(x,g[i],p),marker="o")
# plt.plot(alpha,y)
# plt.show()
