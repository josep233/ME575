import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

#==================================================================================================================================
# def f(x,count):
#     b = 1.5
#     fun = x[0]**2 + x[1]**2 + b * x[0] * x[1]
#     count = count + 1
#     return fun, count
# def ff(x):
#     b = 1.5
#     return x[0]**2 + x[1]**2 + b * x[0] * x[1]
# def f(x,count):
#     fun = (1-x[0])**2 + 100 * (x[1] - x[0]**2)**2
#     count = count + 1
#     return fun, count
# def ff(x):
#     return (1-x[0])**2 + 100 * (x[1] - x[0]**2)**2
def f(x,count):
    fun = (1-x[0])**2 + (1-x[1])**2 + 0.5*(2*x[1]-x[0]**2)**2
    count = count + 1
    return fun, count
def ff(x):
    return (1-x[0])**2 + (1-x[1])**2 + 0.5*(2*x[1]-x[0]**2)**2

# def fp(x):
#     b = 1.5
#     grad1 = 2 * x[0] + b * x[1]
#     grad2 = 2 * x[1] + b * x[0]
#     grad = np.array([grad1,grad2])
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

def phi(x,alpha,p,count):
    fun,count = f(x + alpha * p,count)
    return fun, count

def phip(x,alpha,p):
    return np.dot(fp(x + alpha * p),p)
#==================================================================================================================================
def interpolate(x,alpha1,alpha2,p,count):
    count = phi(x,alpha1,p,count)[1]
    beta1 = phip(x,alpha1,p) + phip(x,alpha2,p) - 3 * ((phi(x,alpha1,p,count)[0] - phi(x,alpha2,p,count)[0])/(alpha1 - alpha2))
    beta2 = np.sign(alpha2 - alpha1) * np.sqrt(beta1**2 - phip(x,alpha1,p) * phip(x,alpha2,p))
    alpha_s = alpha2 - (alpha2 - alpha1) * ((phip(x,alpha2,p) + beta2 - beta1)/(phip(x,alpha2,p) - phip(x,alpha1,p) + 2 * beta2))
    return alpha_s,count
#==================================================================================================================================
def pinpoint(alphalow,alphahigh,x,p,phi0,phip0,mu1,mu2,philow,count):
    k = 0
    g = alphalow
    while True:
        if k < 100:
            alpha_p,count = interpolate(x,alphalow,alphahigh,p,count)
        else:
            alpha_p = (alphalow + alphahigh) / 2

        phi_p = phi(x,alpha_p,p,count)[0]
        if (phi_p > (phi0 + mu1 * alpha_p * phip0)) or (phi_p > philow):
            alphahigh = alpha_p
        else:
            phip_p = phip(x,alpha_p,p)
            if abs(phip_p) <= -mu2 * phip0:
                alpha_s = alpha_p
                g = np.append(g,alpha_s)
                return alpha_p, g, count
            elif phip_p * (alphahigh - alphalow) >= 0:
                alphahigh = alphalow
            alphalow = alpha_p
        k = k + 1
#==================================================================================================================================
def bracketing(x,alphainit,phi0,phip0,p,mu1,mu2,sigma,count):
    alpha1 = 0
    alpha2 = alphainit
    gg = alpha1
    phi1 = phi0
    phip1 = phip0
    first = True
    while True:
        phi2 = phi(x,alpha2,p,count)[0]
        phip2 = phip(x,alpha2,p)
        if (phi2 > phi0 + (mu1 * alpha2 * phip0)) or ((first != True) and (phi2 > phi1)):
            alpha_s, g, count = pinpoint(alpha1,alpha2,x,p,phi0,phip0,mu1,mu2,phi1,count)
            gg = np.append(gg,alpha_s)
            return alpha_s, gg, count
        if abs(phip2) <= -mu2 * phip0:
            alpha_s = alpha2
            gg = np.append(gg,alpha_s)
            return alpha_s, gg, count
        elif phip2 >= 0:
            alpha_s, g, count = pinpoint(alpha2,alpha1,x,p,phi0,phip0,mu1,mu2,phi2,count)
            gg = np.append(gg,alpha_s)
            return alpha_s, gg, count
        else:
            alpha1 = alpha2
            alpha2 = sigma * alpha2
        first = False
#==================================================================================================================================
def alpha_estimate(x_current,x_past,alpha_past,p_current,p_past):
    alpha_current = alpha_past * np.dot(fp(x_past),p_past) / np.dot(fp(x_current),p_current)
    return alpha_current

def steepest(x_current,tau,alpha_past,phi0,phip0,mu1,mu2,sigma):
    k = 0
    count = 1
    norm = np.linalg.norm(fp(x_current))
    normf = 1
    x_past = x_current
    x_list = x_past
    while normf > tau:
        p_current = - fp(x_current) / np.linalg.norm(fp(x_current))
        p_past = - fp(x_past) / np.linalg.norm(fp(x_past))
        alpha_current = alpha_estimate(x_current,x_past,alpha_past,p_current,p_past)
        alpha_p, g, count = bracketing(x_current,alpha_current,phi0,phip0,p_current,mu1,mu2,sigma,count)
        x_past = x_current
        x_current = x_past + alpha_p * p_current
        normf = np.linalg.norm(fp(x_current))
        x_list = np.append(x_list,x_current)
        norm = np.append(norm,normf)
        k = k + 1
        if abs(f(x_past,count)[0] - f(x_current,count)[0]) < 1E-3:
            break
    x_list = np.reshape(x_list,(k+1,2))
    return norm, x_list, k, count
#==================================================================================================================================
def BFGS(tau,x,phi0,phip0,mu1,mu2):
    k = 0
    count = 1
    x0 = x.copy()
    x_list = x.copy()
    alpha_p = 1
    norm = np.linalg.norm(fp(x_current))
    normf = 1
    reset = False
    while normf > tau:
        if k == 0 or reset == True:
            V0 = np.dot((1 / np.linalg.norm(fp(x))),np.eye(len(fp(x))))
            V = V0
            sigma = 2
        else:
            s = x - x0
            y = fp(x) - fp(x0)
            sigma = 1 / (np.dot(np.transpose(s),y))
            V = (np.eye(len(fp(x))) - sigma * np.matmul(s,np.transpose(y))) * V0 * (np.eye(len(fp(x))) - sigma * y * np.transpose(s)) + sigma * s * np.transpose(s)
        p = -V @ fp(x)
        if abs(fp(x) @ p) > 1E-6:
            reset = True
        alpha_p, g, count = bracketing(x,alpha_p,phi0,phip0,p,mu1,mu2,sigma,count)
        x0 = x
        x = x0 + alpha_p * p
        x_list = np.append(x_list,x)
        normf = np.linalg.norm(fp(x))
        norm = np.append(norm,normf)
        k = k + 1
        if abs(f(x0,count)[0] - f(x,count)[0]) < 1E-3:
            break
    x_list = np.reshape(x_list,(k+1,2))
    return norm,x_list,k,count
#==================================================================================================================================

x_current = np.array([-1,0])
p0 = -fp(x_current) / np.linalg.norm(fp(x_current))
tau = 1E-6
alpha_past = 0.1
count = 1
phi0 = phi(x_current,0,p0,count)[0]
phip0 = phip(x_current,0,p0)
mu1 = 0.001
mu2 = 0.01
sigma = 2


norm1, x_list1, iterations1, count1 = steepest(x_current,tau,alpha_past,phi0,phip0,mu1,mu2,sigma)
norm2, x_list2, iterations2, count2 = BFGS(tau,x_current,phi0,phip0,mu1,mu2)
print("feval:",count1, count2)
print("iter:",iterations1, iterations2)

per = minimize(ff,x_current)
print(per)

plt.figure()
plt.plot(np.linspace(0,iterations1,iterations1+1),norm1,"r-",markersize=3)
plt.plot(np.linspace(0,iterations2,iterations2+1),norm2,"b-",markersize=3)
plt.yscale("log")
plt.xlabel("Number of Iterations")
plt.ylabel("Gradient")
plt.title("Convergence - Bean")
plt.legend(["Steepest Descent","Quasi Newton"])
plt.show()




# def fun(x1,x2):
#     fun = np.zeros([len(x1),len(x2)])
#     for i in range(len(x1)):
#         for j in range(len(x2)):
#             fun[i,j] = f([x1[i],x2[j]],count)[0]
#     return fun
# x1 = np.linspace(-2,2,100)
# x2 = np.linspace(-2,2,100)
# plt.figure()
# plt.contour(x1,x2,np.transpose(fun(x1,x2)),100)
# plt.plot(x_list1[:,0],x_list1[:,1],"ro-",markersize=3)
# plt.plot(x_list2[:,0],x_list2[:,1],"bo-",markersize=3)
# plt.axis('equal')
# plt.show()