import fea as s
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime, NonlinearConstraint
from jax import grad

def f(x):
    al = s.truss(x)
    return al

def mass(x):
    mass = f(x)[0]
    return mass

def stress(x):
    stress = f(x)[1]
    return stress

#define first inequality: stresses
def g1(x,i):
    stressa = 25 * 10 **3
    stressb = 75 * 10 **3
    if i != 8:
        if s.truss(x)[1][i] > 0:
            con1 =  stressa - s.truss(x)[1][i]
        else:
            con1 = stressa + s.truss(x)[1][i]
    else:
        if s.truss(x)[1][i] >= 0:
            con1 = stressb - s.truss(x)[1][i]
        else:
            con1 = stressb + s.truss(x)[1][i]
    return con1

#define second inequality: minimum CSA
def g2(x,i):
    con2 = x[i] - .1
    return con2

#initial guess
A0 = np.ones([10,1]) * 10

#create constraints dict
cons = []
for i in range(0,len(A0)):
    cons.append({'type':'ineq','fun':g1,'args':(i,)})
for i in range(0,len(A0)):
    cons.append({'type':'ineq','fun':g2,'args':(i,)})

def fd1(A0,z):
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
    Jmass = Jmass.ravel()
    J = Jmass[z,:]
    Jstress = Jstress.ravel()
    print("mass jac: ",np.shape(Jmass))
    return J
def fd2(A0,z):
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
    Jmass = Jmass.ravel()
    Jstress = Jstress.ravel()
    Jstress = np.reshape(Jstress,(10,10))
    J = Jstress[:,z]
    print("stress jac: ",np.shape(Jstress))
    return J

def cs1(A0):
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
    return Jmass
def cs2(A0):
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
        Jstress = np.reshape(Jstress,(10,10))
    return Jstress

# def ad(A0):
#     Jmass = grad(mass,A0)
#     Jstress = grad(stress,A0)
#     return Jmass



#calllback function creation for tracking convergence
Nfeval = 1
fe = []
mas = []
jac = 0
cjac = 0
def callb(A0):
    global Nfeval
    # global Jmass
    # global Jall
    # global Jstress
    fe.append(Nfeval)
    mas.append(mass(A0))
    # Jmass = (fd1(A0))
    # Jstress = (fd2(A0))
    # Jall = np.reshape(np.append(Jmass,Jstress),(11,10))
    # jac1 = (approx_fprime(A0, mass, 1E-8))
    # jac2 = (approx_fprime(A0, stress, 1E-8))
    # masserror = np.append(masserror,np.max(abs((Jmass - jac1)/jac1)))
    # stresserror = np.append(stresserror,np.max(abs((Jstress - jac2)/jac2)))
    print("function evaluation: ",Nfeval)
    # print("calculated gradient: ",Jstress)
    # print("actual gradient: ",jac2)
    Nfeval += 1

#create constraints dict
cons2 = []
for i in range(0,len(A0)):
    cons2 = np.append(cons2,NonlinearConstraint(lambda x: g2(x,i),lb=0,ub=np.inf,jac=lambda x: fd1(x,i)))
for i in range(0,len(A0)):
    cons2 = np.append(cons2,NonlinearConstraint(lambda x: g1(x,i),lb=0,ub=np.inf,jac=lambda x: fd2(x,i)))
    cons2 = np.array([cons2])

print(cons2)

#vanilla optimization
ans = minimize(mass,A0,constraints = cons2.all(),callback=callb,options={'maxiter':100})
print(ans)