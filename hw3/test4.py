
#import needed libraries
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import jax.numpy as np
import numpy as npp
from math import sin, cos, sqrt, pi

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
    K = np.zeros((DOF*n, DOF*n))
    S = np.zeros((nbar, DOF*n))

    for i in range(nbar):  # loop through each bar

        # compute submatrix for each element
        Ksub, Ssub = bar(E[i], A[i], L[i], phi[i])

        # insert submatrix into global matrix
        idx = node2idx([start[i], finish[i]], DOF)  # pass in the starting and ending node number for this element
        K = K.at[np.ix_(idx, idx)].add(Ksub)
        S = S.at[i, idx].set(Ssub)
    # applied loads
    F = np.zeros((n*DOF, 1))

    for i in range(n):
        idx = node2idx([i+1], DOF)  # add 1 b.c. made indexing 1-based for convenience
        F = F.at[idx[0]].set(Fx[i])
        F = F.at[idx[1]].set(Fy[i])


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

import jax.numpy as np
from scipy.optimize import minimize
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import NonlinearConstraint
from scipy.optimize._numdiff import approx_derivative
import jax

#define starting functions
def f(x):
    al = truss(x)
    return al
def mass(x):
    mass = f(x)[0]
    return mass
def stress(x):
    stress = f(x)[1]
    return stress

#define constraint
def g(x):
    con = np.zeros((2*len(x),1))
    for i in range(0,len(x)):
        con = con.at[i].set(x[i] - .1)
    for i in range(0,len(x)):
        stressa = 25 * 10 **3
        stressb = 75 * 10 **3
        if i != 8:
            if stress(x)[i] > 0:
                con = con.at[i+len(x)].set(stressa - stress(x)[i])
            else:
                con = con.at[i+len(x)].set(stressa + stress(x)[i])
        else:
            if stress(x)[i] >= 0:
                con = con.at[i+len(x)].set(stressb - stress(x)[i])
            else:
                con = con.at[i+len(x)].set(stressb + stress(x)[i])
    return con
#======================ALGORITHMIC DIFFERENTIATION================================
def dg(A0):
    Jg = jax.jacfwd(g)
    Jg = Jg(A0)
    Jg = np.reshape(Jg,(20,10))
    return Jg
def df(A0):
    Jf = jax.jacfwd(mass)
    Jf = Jf(A0)
    return Jf
#=================================================================================

def obj(A0):
    return mass(A0),df(A0)
cons2 = NonlinearConstraint(g,lb=0,ub=np.inf,jac=dg)

#initial guess
A0 = np.ones((10,1)) * 10

#calllback function creation for tracking convergence
Nfeval = 1
jacerror = np.array((0))
conv = np.array((0))
def callb(A0):
    global Nfeval
    global jacerror
    global conv
    actualjac = approx_derivative(mass,A0)
    calcjac = df(A0)
    jacerror = np.append(jacerror,100 * np.max(abs((calcjac - actualjac)/actualjac)))
    conv = np.append(conv,100 * abs((mass(A0) - 1497.0)/1497.0))
    print("actual jac: ",actualjac)
    print("calculated jac: ",calcjac)
    print("function value: ",mass(A0))
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
plt.title('AD Relative Error')
plt.figure(2)
plt.plot(x,conv)
plt.ylabel("Percent Relative Error from M_final (%)")
plt.xlabel("Number of Function Evaluations")
plt.title("Convergence Plot - AD")
plt.show()

