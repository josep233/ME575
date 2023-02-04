import numpy as np
from matplotlib import pyplot as plt
import funs as f

def interpolate(x,alpha1,alpha2,p):
    if abs(alpha1 - alpha2) > 1E-10:
        beta1 = f.phip(x,alpha1,p) + f.phip(x,alpha2,p) - 3 * ((f.phi(x,alpha1,p) - f.phi(x,alpha2,p))/(alpha1 - alpha2))
        if beta1**2 - f.phip(x,alpha1,p) * f.phip(x,alpha2,p) > 0:
            beta2 = np.sign(alpha2 - alpha1) * np.sqrt(beta1**2 - f.phip(x,alpha1,p) * f.phip(x,alpha2,p))
            alpha_s = alpha2 - (alpha2 - alpha1) * ((f.phip(x,alpha2,p) + beta2 - beta1)/(f.phip(x,alpha2,p) - f.phip(x,alpha1,p) + 2 * beta2))
            print(alpha_s)
            return alpha_s
        else:
            alpha_s = (alpha1 + alpha2) / 2
            return alpha_s
    else:
        alpha_s = (alpha1 + alpha2) / 2
        return alpha_s

# x = np.array([3.5,5])
# p = - fp(x) / np.linalg.norm(fp(x))
# alpha1 = 0
# alpha2 = 10
# alpha_s = interpolate(x,alpha1,alpha2,p)
# new_point = phi(x,alpha_s,p)
# print(alpha_s)

# plt.figure()
# alpha = np.linspace(0,10,100)
# y = np.zeros([len(alpha),1])
# plt.plot(alpha_s,new_point,marker="o")
# for i in range(0,len(alpha)):
#     y[i] = phi(x,alpha[i],p)
# plt.plot(alpha,y)
# plt.show()