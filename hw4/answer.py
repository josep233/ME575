import ABQScript as a
import numpy as np
from scipy.optimize import minimize, approx_fprime
import subprocess
from scipy.optimize._numdiff import approx_derivative

const = 9

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
    ans = float(ans)*10**-(const)
    print("function evaluation: ",Nfeval)
    print("eigenvalue: ",ans*10**(const))
    Nfeval += 1
    return ans

def con1(init):
       return ((350*10**(-const)) - init[0])
def con2(init):
       return (init[0])*10**(-const)
def con3(init):
       return ((2790*10**(-const)) - init[1])
def con4(init):
       return (init[1])*10**(-const)
def con5(init):
       return ((2800*10**(-const)) - init[2])
def con6(init):
       return (init[2])*10**(-const)
def con7(init):
       return ObjectiveFunction(init) - 15.0*10**(-const)


cons = [
        {'type':'ineq','fun':con1},
        {'type':'ineq','fun':con2},
        {'type':'ineq','fun':con3},
        {'type':'ineq','fun':con4},
        {'type':'ineq','fun':con5},
        {'type':'ineq','fun':con6},
        {'type':'ineq','fun':con7},
        ]

init = np.array([20,100,100])*10**-const
for i in range(0,10):
       res = minimize(ObjectiveFunction,init,constraints=cons,options={"maxiter":1})
       init = res.x
       print("|||||||||||||||||||||jac: ",res.jac)