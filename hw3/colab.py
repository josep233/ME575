#import needed libraries
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import autograd.numpy as np
from math import sin, cos, sqrt, pi
from scipy.optimize import approx_fprime
from jax import grad
import jax.numpy as np

def truss(A):
    """Computes mass and stress for the 10-bar truss problem
    Parameters
    ----------
    A : ndarray of length nbar
        cross-sectional areas of each bar
        see image in book for number order if needed
    Outputs
    -------
    mass : float
        mass of the entire structure
    stress : ndarray of length nbar
        stress in each bar
    """

    # --- specific truss setup -----
    P = 1e5  # applied loads
    Ls = 360.0  # length of sides
    Ld = sqrt(360**2 * 2)  # length of diagonals

    start = [5, 3, 6, 4, 4, 2, 5, 6, 3, 4]
    finish = [3, 1, 4, 2, 3, 1, 4, 3, 2, 1]
    phi = np.array([0, 0, 0, 0, 90, 90, -45, 45, -45, 45])*pi/180
    L = np.array([Ls, Ls, Ls, Ls, Ls, Ls, Ld, Ld, Ld, Ld])

    nbar = len(A)  # number of bars
    E = 1e7*np.ones(nbar)  # modulus of elasticity
    rho = 0.1*np.ones(nbar)  # material density

    Fx = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    Fy = np.array([0.0, -P, 0.0, -P, 0.0, 0.0])
    rigid = [False, False, False, False, True, True]
    # ------------------

    n = len(Fx)  # number of nodes
    DOF = 2  # number of degrees of freedom

    # mass
    mass = np.sum(rho*A*L)

    # stiffness and stress matrices
    # K = np.zeros((DOF*n, DOF*n),dtype=complex)
    # S = np.zeros((nbar, DOF*n),dtype=complex)
    K = np.zeros((DOF*n, DOF*n))
    S = np.zeros((nbar, DOF*n))

    for i in range(nbar):  # loop through each bar

        # compute submatrix for each element
        Ksub, Ssub = bar(E[i], A[i], L[i], phi[i])

        # insert submatrix into global matrix
        idx = node2idx([start[i], finish[i]], DOF)  # pass in the starting and ending node number for this element
        K.at[np.ix_(idx, idx)].add(Ksub)
        S.at[i, idx].set(Ssub)
    # applied loads
    F = np.zeros((n*DOF, 1))

    for i in range(n):
        idx = node2idx([i+1], DOF)  # add 1 b.c. made indexing 1-based for convenience
        F.at[idx[0]].set(Fx[i])
        F.at[idx[1]].set(Fy[i])


    # boundary condition
    idx = [i+1 for i, val in enumerate(rigid) if val] # add 1 b.c. made indexing 1-based for convenience
    remove = node2idx(idx, DOF)

    K = np.delete(K, remove, axis=0)
    K = np.delete(K, remove, axis=1)
    F = np.delete(F, remove, axis=0)
    S = np.delete(S, remove, axis=1)

    # solve for deflections
    d = np.linalg.solve(K, F)

    # compute stress
    stress = np.dot(S, d).reshape(nbar)

    return mass, stress



def bar(E, A, L, phi):
    """Computes the stiffness and stress matrix for one element
    Parameters
    ----------
    E : float
        modulus of elasticity
    A : float
        cross-sectional area
    L : float
        length of element
    phi : float
        orientation of element
    Outputs
    -------
    K : 4 x 4 ndarray
        stiffness matrix
    S : 1 x 4 ndarray
        stress matrix
    """

    # rename
    c = cos(phi)
    s = sin(phi)

    # stiffness matrix
    k0 = np.array([[c**2, c*s], [c*s, s**2]])
    k1 = np.hstack([k0, -k0])
    K = E*A/L*np.vstack([k1, -k1])

    # stress matrix
    S = E/L*np.array([-c, -s, c, s])

    return K, S



def node2idx(node, DOF):
    """Computes the appropriate indices in the global matrix for
    the corresponding node numbers.  You pass in the number of the node
    (either as a scalar or an array of locations), and the degrees of
    freedom per node and it returns the corresponding indices in
    the global matrices
    """

    idx = np.array([], dtype=int)

    for i in range(len(node)):

        n = node[i]
        start = DOF*(n-1)
        finish = DOF*n

        idx = np.concatenate((idx, np.arange(start, finish, dtype=int)))

    return idx

#=============================================================================================================
def f(x):
    al = truss(x)
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
        if truss(x)[1][i] > 0:
            con1 =  stressa - truss(x)[1][i]
        else:
            con1 = stressa + truss(x)[1][i]
    else:
        if truss(x)[1][i] >= 0:
            con1 = stressb - truss(x)[1][i]
        else:
            con1 = stressb + truss(x)[1][i]
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

def fd(A0):
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
    return Jmass, Jstress

def cs(A0):
    iA0 = A0.copy()
    iA0 = iA0.astype('complex')
    h = 1E-200
    Jmass = np.zeros([1,len(iA0)])
    Jstress =  np.zeros([len(iA0),len(stress(iA0))])
    for j in range(0,len(iA0)):
        iA0[j] = complex(iA0[j],0) + complex(0,h)
        print(A0[j])
        mass_plus = mass(iA0)
        stress_plus = stress(iA0)
        Jmass[0][j] = np.imag(mass_plus) / h
        Jstress[:,j] = np.imag(stress_plus) / h
        iA0[j] = complex(iA0[j],0) - complex(0,h)
    return Jmass, Jstress

def ad(A0):
    Jmass = grad(mass)
    ans = Jmass(A0)
    return ans

#callback function creation for tracking convergence
Nfeval = 1
fe = []
mas = []
# jac = 0
cjac = 0
def callb(A0):
    global Nfeval
    fe.append(Nfeval)
    mas.append(mass(A0))
    cjac = ad(A0)
    jac = approx_fprime(A0, mass, 1E-8)
    print("function evaluation: ",Nfeval)
    print("calculated gradient: ",cjac)
    print("actual gradient: ",jac)
    Nfeval += 1


#vanilla optimization
ans = minimize(mass,A0,constraints = cons,callback=callb,options={'maxiter':2})
