import numpy as np
from matplotlib import pyplot as plt


def f(x):
    b = 0
    return x[0]**2 + x[1]**2 + b * x[0] * x[1]

def fp(x):
    b = 0
    grad1 = 2 * x[0] + b * x[1]
    grad2 = 2 * x[1] + b * x[0]
    grad = np.array([grad1,grad2])
    return grad

def phi(x,alpha,p):
    return f(x + alpha * p)

def phip(x,alpha,p):
    return np.dot(fp(x + alpha * p),p)

def interpolate(x,alpha1,alpha2,p):
    if abs(alpha1 - alpha2) > 1E-10:
        beta1 = phip(x,alpha1,p) + phip(x,alpha2,p) - 3 * ((phi(x,alpha1,p) - phi(x,alpha2,p))/(alpha1 - alpha2))
        beta2 = np.sign(alpha2 - alpha1) * np.sqrt(beta1**2 - phip(x,alpha1,p) * phip(x,alpha2,p))
        alpha_s = alpha2 - (alpha2 - alpha1) * ((phip(x,alpha2,p) + beta2 - beta1)/(phip(x,alpha2,p) - phip(x,alpha1,p) + 2 * beta2))
    else:
        alpha_s = (alpha1 + alpha2) / 2
    return alpha_s

# x = np.array([3.5,5])
# p = - fp(x) / np.linalg.norm(fp(x))
# alpha1 = 6.103277807866850
# alpha2 = 6.103277807866852
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