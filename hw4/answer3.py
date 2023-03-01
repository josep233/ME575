import ABQScript as a
import numpy as np
from scipy.optimize import minimize, approx_fprime, NonlinearConstraint
import subprocess
from scipy.optimize._numdiff import approx_derivative

const = 8

Nfeval = 1
def ObjectiveFunction(init):
    global Nfeval
    pithE1 = 26
    pithE2 = 26
    pithE3 = 10
    pithNu12 = 0.3
    pithNu13 = 0.05
    pithNu23 = 0.05
    pithG12 = 10
    pithG13 = 11
    pithG23 = 11
    rindE1 = 850
    rindE2 = 850
    rindE3 = 12995
    rindNu12 = 0.3
    rindNu13 = 0.05
    rindNu23 = 0.05
    
    pithprops = (pithE1,pithE2,pithE3,pithNu12,pithNu13,pithNu23,pithG12,pithG13,pithG23,)
    rindprops = (rindE1,rindE2,rindE3,rindNu12,rindNu13,rindNu23,init[0]*10**const,init[1]*10**const,init[2]*10**const,)
    a.script(pithprops,rindprops)
    subprocess.run("abaqus cae noGUI=C:\\Users\\Joseph\\Desktop\\Temp\\test.py", shell=True)
    file = open('C:\\Users\\Joseph\\Desktop\\Temp\\test.txt',"r")
    ans = np.double(file.readline())
    ans = float(ans)*10**(const)
    print("function evaluation: ",Nfeval)
    print("eigenvalue: ",ans*10**(-const))
    Nfeval += 1
    return ans


def constraintfuns(x):
       con = np.zeros((6,1))
       con[0] = 3000 - x[0]*10**const
       con[1] = x[0]
       con[2] = 3000 - x[1]*10**const
       con[3] = x[1]
       con[4] = 3000 - x[2]*10**const
       con[5] = x[2]
       return con.ravel()

def endcon(x):
      return ObjectiveFunction(x) - 40

constraints = {'type':'eq','fun':endcon}

init = np.array((2000,2500,2000))*10**-const
res = minimize(ObjectiveFunction,init,constraints=constraints,bounds=((326*10**-const,2000*10**-const),(389*10**-const,2500*10**-const),(389*10**-const,2000*10**-const),))
print(res)