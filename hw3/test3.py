import fea as s
import numpy as np
from scipy.optimize import minimize
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime, NonlinearConstraint
# from jax import grad

#define starting functions
def f(x):
    al = s.truss(x)
    return al
def mass(x):
    mass = f(x)[0]
    return mass
def stress(x):
    stress = f(x)[1]
    return stress

#define constraint
def g(x):
    con = np.zeros([2*len(x),1],dtype='complex')
    for i in range(0,len(x)):
        con[i] = x[i] - .1
    for i in range(0,len(x)):
        stressa = 25 * 10 **3
        stressb = 75 * 10 **3
        if i != 8:
            if stress(x)[i] > 0:
                con[i+len(x)] =  stressa - stress(x)[i]
            else:
                con[i+len(x)] = stressa + stress(x)[i]
        else:
            if stress(x)[i] >= 0:
                con[i+len(x)] = stressb - stress(x)[i]
            else:
                con[i+len(x)] = stressb + stress(x)[i]
    return con.ravel()

#====================FINITE DIFFERENCE===========================================
def dg(A0):
    h = 1E-6
    g0 = g(A0)
    Jg = np.zeros([2*len(A0),len(A0)])
    for j in range(0,len(A0)):
        delta_x = h * (1 + abs(A0[j]))
        A0[j] = A0[j] + delta_x
        gplus = g(A0)
        Jg[:,j] = ((gplus - g0) / delta_x).ravel()
        A0[j] = A0[j] - delta_x
    return Jg
def df(A0):
    h = 1E-6
    f0 = mass(A0)
    Jf = np.zeros([len(A0),1])
    for j in range(0,len(A0)):
        delta_x = h * (1 + abs(A0[j]))
        A0[j] = A0[j] + delta_x
        fplus = mass(A0)
        Jf[j] = ((fplus - f0) / delta_x)
        A0[j] = A0[j] - delta_x
    return Jf
#====================COMPLEX STEP===============================================
# def dg(A0):
#     iA0 = A0.copy()
#     iA0 = iA0.astype('complex')
#     h = 1E-200
#     Jg = np.zeros([2*len(A0),len(A0)],dtype='complex')
#     for j in range(0,len(iA0)):
#         iA0[j] = complex(iA0[j],0) + complex(0,h)
#         gplus = g(iA0)
#         Jg[:,j] = np.imag(gplus) / h
#         iA0[j] = complex(iA0[j],0) - complex(0,h)
#     return Jg
# def df(A0):
#     iA0 = A0.copy()
#     iA0 = iA0.astype('complex')
#     h = 1E-200
#     Jf = np.zeros([len(iA0),1])
#     for j in range(0,len(iA0)):
#         iA0[j] = complex(iA0[j],0) + complex(0,h)
#         fplus = mass(iA0)
#         Jf[j] = np.imag(fplus) / h
#         iA0[j] = complex(iA0[j],0) - complex(0,h)
#     return Jf
#=================================================================================

def obj(A0):
    return mass(A0),df(A0)
cons2 = NonlinearConstraint(g,lb=0,ub=np.inf,jac=dg)

#initial guess
A0 = np.ones([10,1]) * 10

#calllback function creation for tracking convergence
Nfeval = 1
jacerror = []
conv = []
def callb(A0):
    global Nfeval
    global jacerror
    global conv
    actualjac = abs(approx_fprime(A0, mass, 1E-8))
    calcjac = abs(df(A0))
    jacerror = np.append(jacerror,100 * np.max(abs((calcjac - actualjac)/actualjac)))
    conv = np.append(conv,100 * abs((mass(A0) - 1497.0)/1497.0))
    print("function evaluation: ",Nfeval)
    Nfeval += 1

# cons = {'type':'ineq','fun':g2}

ans = minimize(obj,A0,constraints = cons2,callback=callb,method='SLSQP',jac=True,options={'maxiter':100})
# ans = minimize(mass,A0,constraints = cons,callback=callb)
print(ans)

x = np.linspace(0,Nfeval,len(conv))

plt.figure(1)
plt.boxplot(jacerror)
plt.ylabel('Maximum Percent Relative Error (%)')
plt.title('Complex Step Relative Error')
plt.figure(2)
plt.plot(x,conv)
plt.ylabel("Percent Relative Error from M_final (%)")
plt.xlabel("Number of Function Evaluations")
plt.title("Convergence Plot - Complex Step")
plt.show()

