import numpy as np
import interpolate as I
from matplotlib import pyplot as plt
import numdifftools as nd
import funs as f

def pinpoint(alphalow,alphahigh,x,p,phi0,phip0,mu1,mu2,philow):
    k = 0
    g = alphalow
    while True:
        if k < 100:
        # if ((f.phip(x,alphalow,p) + f.phip(x,alphahigh,p) - 3 * ((f.phi(x,alphalow,p) - f.phi(x,alphahigh,p))/(alphalow - alphahigh)))**2 - f.phip(x,alphalow,p) * f.phip(x,alphahigh,p) > 0) and (abs(alphalow - alphahigh) < 1E-6):
            alpha_p = I.interpolate(x,alphalow,alphahigh,p)
        else:
            alpha_p = (alphalow + alphahigh) / 2

        phi_p = f.phi(x,alpha_p,p)
        if (phi_p > (phi0 + mu1 * alpha_p * phip0)) or (phi_p > philow):
            alphahigh = alpha_p
        else:
            phip_p = f.phip(x,alpha_p,p)
            if abs(phip_p) <= -mu2 * phip0:
                alpha_s = alpha_p
                g = np.append(g,alpha_s)
                return alpha_p, g
            elif phip_p * (alphahigh - alphalow) >= 0:
                alphahigh = alphalow
            alphalow = alpha_p
        k = k + 1

# x = np.array([-7.5,8])
# p = -f.fp(x) / np.linalg.norm(f.fp(x))
# alphalow = 0
# alphahigh = 0.1
# phi0 = f.phi(x,0,p)
# phip0 = f.phip(x,0,p)
# mu1 = 0.1
# mu2 = 0.2
# philow = f.phi(x,alphalow,p)

# print(alphalow,alphahigh,x,p,phi0,phip0,mu1,mu2,philow)

# alpha_p, g = pinpoint(alphalow,alphahigh,x,p,phi0,phip0,mu1,mu2,philow)

# new_point = f.phi(x,alpha_p,p)
# print(alpha_p)

# plt.figure()
# alpha = np.linspace(0,10,100)
# y = np.zeros([len(alpha),1])
# plt.plot(alpha_p,new_point,marker="o")
# for i in range(0,len(alpha)):
#     y[i] = f.phi(x,alpha[i],p)
# for i in range(0,len(g)):
#     plt.plot(g[i],f.phi(x,g[i],p),marker="o")
# print(len(g))
# plt.plot(alpha,y)
# plt.show()

