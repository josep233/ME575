import fea as s
import numpy as np
from scipy.optimize import minimize
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime, NonlinearConstraint
# from jax import grad

def f(x):
    al = s.truss(x)
    return al

def mass(x):
    mass = f(x)[0]
    return mass

def stress(x):
    stress = f(x)[1]
    return stress

#define first inequality: minimum CSA
def g1(x,i):
    con1 = x[i] - .1
    return con1

#define second inequality: stresses
def g2(x,i):
    stressa = 25 * 10 **3
    stressb = 75 * 10 **3
    if i != 8:
        if f(x)[1][i] > 0:
            con2 =  stressa - f(x)[1][i]
        else:
            con2 = stressa + f(x)[1][i]
    else:
        if f(x)[1][i] >= 0:
            con2 = stressb - f(x)[1][i]
        else:
            con2 = stressb + f(x)[1][i]
    return con2

#initial guess
A0 = np.ones([10,1]) * 0.1

cons = []
for i in range(0,len(A0)):
    cons.append({'type':'ineq','fun':g1,'args':(i,)})
for i in range(0,len(A0)):
    cons.append({'type':'ineq','fun':g2,'args':(i,)})

def fd(A0,z):
    m0 = mass(A0)
    s0 = stress(A0)
    h = 1E-6
    Jmass = np.zeros([1,len(A0)])
    Jstress =  np.zeros([len(A0),len(stress(A0))])
    for j in range(0,len(A0)):
        delta_x = h * (1 + abs(A0[j]))
        A0[j] = A0[j] + delta_x
        mass_plus = mass(A0)
        stress_plus = stress(A0)
        Jmass[0][j] = (mass_plus - m0) / delta_x
        Jstress[:,j] = (stress_plus - s0) / delta_x
        A0[j] = A0[j] - delta_x
    Jmasscut = Jmass.ravel()
    Jstresscut = Jstress[z,:].ravel()
    return Jmasscut,Jstresscut,Jmass,Jstress

def df(A0):
    h = 1E-6
    Jf = np.zeros([1,len(A0)])
    f0 = f(A0)
    for j in range(0,len(A0)):
        delta_x = h * (1 + abs(A0[j]))
        A0[j] = A0[j] + delta_x
        fplus = f(A0)
        Jf[0][j] = (fplus - f0) / delta_x
        A0[j] = A0[j] - delta_x
    return Jf

def fg1(A0)

def cs(A0,z):
    iA0 = A0.copy()
    iA0 = iA0.astype('complex')
    h = 1E-200
    Jmass = np.zeros([1,len(iA0)])
    Jstress =  np.zeros([len(iA0),len(stress(iA0))])
    for j in range(0,len(iA0)):
        iA0[j] = complex(iA0[j],0) + complex(0,h)
        mass_plus = mass(iA0)
        stress_plus = stress(iA0)
        Jmass[0][j] = np.imag(mass_plus) / h
        Jstress[:,j] = np.imag(stress_plus) / h
        iA0[j] = complex(iA0[j],0) - complex(0,h)
    Jmasscut = Jmass.ravel()
    Jstresscut = Jstress[z,:].ravel()
    return Jmasscut,Jstresscut,Jmass,Jstress

#calllback function creation for tracking convergence
Nfeval = 1
fe = []
mas = []
masserror = []
stresserror = []
def callb(A0):
    global Nfeval
    global masserror
    global stresserror
    actualmass = (approx_fprime(A0, mass, 1E-8))
    actualstress = (approx_fprime(A0, stress, 1E-8))
    calcmass = fd(A0,1)[2]
    calcstress = fd(A0,1)[3]
    masserror = np.append(masserror,np.max(abs((calcmass - actualmass)/actualmass)))
    stresserror = np.append(stresserror,np.max(abs((calcstress - actualstress)/actualstress)))
    print("function evaluation: ",Nfeval)
    Nfeval += 1

#create constraints dict
cons2 = []
for i in range(0,len(A0)):
    cons2 = np.append(cons2,NonlinearConstraint(lambda x: g1(x,i),lb=0,ub=np.inf,jac=lambda x: fd(x,i)[0]))
for i in range(0,len(A0)):
    cons2 = np.append(cons2,NonlinearConstraint(lambda x: g2(x,i),lb=0,ub=np.inf,jac=lambda x: fd(x,i)[1]))
    cons2 = np.array([cons2])

#vanilla optimization
# ans = minimize(mass,A0,constraints = cons,callback=callb,options={'maxiter':100})

# opts = scipy.optimize.show_options(solver='minimize')
# print(opts)

ans = minimize(mass,A0,constraints = cons2.any(),callback=callb,method='SLSQP',jac=True)
print(ans)


plt.figure()
plt.boxplot(masserror)
plt.show()
