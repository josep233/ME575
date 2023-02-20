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

def gstress0(x):
    if s.truss(x)[1][0] > 0:
        con = 25 * 10**3 - s.truss(x)[1][0]
    else:
        con = 25 * 10**3 + s.truss(x)[1][0]
    return con
def gstress1(x):
    if s.truss(x)[1][1] > 0:
        con = 25 * 10**3 - s.truss(x)[1][1]
    else:
        con = 25 * 10**3 + s.truss(x)[1][1]
    return con
def gstress2(x):
    if s.truss(x)[1][2] > 0:
        con = 25 * 10**3 - s.truss(x)[1][2]
    else:
        con = 25 * 10**3 + s.truss(x)[1][2]
    return con
def gstress3(x):
    if s.truss(x)[1][3] > 0:
        con = 25 * 10**3 - s.truss(x)[1][3]
    else:
        con = 25 * 10**3 + s.truss(x)[1][3]
    return con
def gstress4(x):
    if s.truss(x)[1][4] > 0:
        con = 25 * 10**3 - s.truss(x)[1][4]
    else:
        con = 25 * 10**3 + s.truss(x)[1][4]
    return con
def gstress5(x):
    if s.truss(x)[1][5] > 0:
        con = 25 * 10**3 - s.truss(x)[1][5]
    else:
        con = 25 * 10**3 + s.truss(x)[1][5]
    return con
def gstress6(x):
    if s.truss(x)[1][6] > 0:
        con = 25 * 10**3 - s.truss(x)[1][6]
    else:
        con = 25 * 10**3 + s.truss(x)[1][6]
    return con
def gstress7(x):
    if s.truss(x)[1][7] > 0:
        con = 25 * 10**3 - s.truss(x)[1][7]
    else:
        con = 25 * 10**3 + s.truss(x)[1][7]
    return con
def gstress8(x):
    if s.truss(x)[1][8] > 0:
        con = 75 * 10 **3 - s.truss(x)[1][8]
    else:
        con = 75 * 10 **3 + s.truss(x)[1][8]
    return con
def gstress9(x):
    if s.truss(x)[1][9] > 0:
        con = 25 * 10**3 - s.truss(x)[1][9]
    else:
        con = 25 * 10**3 + s.truss(x)[1][9]
    return con

def gmass0(x):
    return x[0] - .1
def gmass1(x):
    return x[1] - .1
def gmass2(x):
    return x[2] - .1
def gmass3(x):
    return x[3] - .1
def gmass4(x):
    return x[4] - .1
def gmass5(x):
    return x[5] - .1
def gmass6(x):
    return x[6] - .1
def gmass7(x):
    return x[7] - .1
def gmass8(x):
    return x[8] - .1
def gmass9(x):
    return x[9] - .1

#initial guess
A0 = np.ones([10,1]) * 10

constraints = [gstress0,gstress1,gstress2,gstress3,gstress4,gstress5,gstress6,gstress7,gstress8,gstress9,gmass0,gmass1,gmass2,gmass3,gmass4,gmass5,gmass6,gmass7,gmass8,gmass9]


def fd1(A0):
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
    Jmass = np.reshape(Jmass,(10,1))
    Jmass = np.tile(Jmass,(10,1))
    Jstress = Jstress.ravel()
    Jall = np.transpose(np.reshape(np.append(Jmass,Jstress),(11,20)))
    print("all jac shape: ",np.shape(Jall))
    return Jall
def fd2(A0):
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
    print("stress jac shape: ",np.shape(Jstress))
    return Jstress

Nfeval = 1
fe = []
mas = []
jac = 0
cjac = 0
masserror = []
stresserror = []
def callb(A0):
    global Nfeval
    global masserror
    global stresserror
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

def stresses(x):
    stress0 = gstress0(x)
    stress1 = gstress1(x)
    stress2 = gstress2(x)
    stress3 = gstress3(x)
    stress4 = gstress4(x)
    stress5 = gstress5(x)
    stress6 = gstress6(x)
    stress7 = gstress7(x)
    stress8 = gstress8(x)
    stress9 = gstress9(x)
    stress = [stress0,stress1,stress2,stress3,stress4,stress5,stress6,stress7,stress8,stress9]
    # print("constraint shape: ", np.shape(stress))
    return stress
def masses(x):
    mass0 = gmass0(x)
    mass1 = gmass1(x)
    mass2 = gmass2(x)
    mass3 = gmass3(x)
    mass4 = gmass4(x)
    mass5 = gmass5(x)
    mass6 = gmass6(x)
    mass7 = gmass7(x)
    mass8 = gmass8(x)
    mass9 = gmass9(x)
    mass = [mass0,mass1,mass2,mass3,mass4,mass5,mass6,mass7,mass8,mass9]
    # print("constraint shape: ", np.shape(mass))
    return mass
def all(x):
    stress = stresses(x)
    mass = masses(x)
    all = np.append(mass,stress)
    all = np.reshape(all,(10,2))
    print("constraint shape: ", np.shape(all))
    return all    

# # cons2 = [NonlinearConstraint(all,0,np.inf,jac=fd1),
# #         ]
cons2 = NonlinearConstraint(all,0,np.inf,jac=fd1)

ans = minimize(mass,A0,constraints = cons2,callback=callb,options={'maxiter':100})
print(ans)

