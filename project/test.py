import Constraints as c
import ObjectiveFunction as f
import RindE3 as r
import numpy as np
from scipy.optimize import minimize
import subprocess
import ABQScript as a

stalk = 7
pithE1 = 0.026 * 1000
pithE2 = pithE1
pithE3 = 0.11 * 1000
pithNu12 = 0.3
pithNu13 = 0.05
pithNu23 = pithNu13
pithG12 = pithE1 / (2*(1+pithNu12))
pithG13 = 0.1 * pithE3
pithG23 = pithG13
rindE1 = 0.85 * 1000
rindE2 = rindE1
rindE3 = r.rinde3(stalk)
rindNu12 = pithNu12
rindNu13 = pithNu13
rindNu23 = pithNu23
rindG12 = rindE1 / (2*(1+rindNu12))
rindG13 = 0.1 * rindE3
rindG23 = rindG13


Nfeval = 1
def callb():
    global Nfeval
    print(Nfeval)
    Nfeval += 1

def ObjectiveFunction(init):
    pithE1 = init[0]
    pithE2 = init[1]
    pithE3 = init[2]
    pithNu12 = init[3]
    pithNu13 = init[4]
    pithNu23 = init[5]
    pithG12 = init[6]
    pithG13 = init[7]
    pithG23 = init[8]
    rindE1 = init[9]
    rindE2 = init[10]
    rindE3 = init[11]
    rindNu12 = init[12]
    rindNu13 = init[13]
    rindNu23 = init[14]
    rindG12 = init[15]
    rindG13 = init[16]
    rindG23 = init[17]
    
    pithprops = (pithE1,pithE2,pithE3,pithNu12,pithNu13,pithNu23,pithG12,pithG13,pithG23,)
    rindprops = (rindE1,rindE2,rindE3,rindNu12,rindNu13,rindNu23,rindG12,rindG13,rindG23,)
    a.script(stalk,pithprops,rindprops)
    subprocess.run("abaqus cae noGUI=C:\\Users\\Joseph\\Desktop\\Temp\\test.py", shell=True)
    file = open('C:\\Users\\Joseph\\Desktop\\Temp\\test.txt',"r")
    ans = file.readline()
    return ans

def constraints(init):
    pithE1 = init[0]
    pithE2 = init[1]
    pithE3 = init[2]
    pithNu12 = init[3]
    pithNu13 = init[4]
    pithNu23 = init[5]
    pithG12 = init[6]
    pithG13 = init[7]
    pithG23 = init[8]
    rindE1 = init[9]
    rindE2 = init[10]
    rindE3 = init[11]
    rindNu12 = init[12]
    rindNu13 = init[13]
    rindNu23 = init[14]
    rindG12 = init[15]
    rindG13 = init[16]
    rindG23 = init[17]
    
    rindG13_low_constant = 0.03
    rindG13_high_constant = 0.21
    rindE1_avg = 0.85 * 1000
    rindE1_mu = 0.39 * 1000
    rindNu12_low = 0.2
    rindNu12_high = 0.45
    rindNu13_low = 0.009
    rindNu13_high = 0.086
    pithG13_low_constant = 0.03
    pithG13_high_constant = 0.21
    pithE1_avg = 0.026 * 1000
    pithE1_mu = 0.01 * 1000
    pithNu12_low = 0.2
    pithNu12_high = 0.45
    pithNu13_low = 0.009
    pithNu13_high = 0.086
    pithE3_low = 0.06 * 1000
    pithE3_high = 0.18 * 1000 

    pithNu31 = pithNu13 * (pithE3 / pithE1)
    pithNu32 = pithNu23 * (pithE3 / pithE2)
    pithNu21 = pithNu12 * (pithE2 / pithE1)
    rindNu31 = rindNu13 * (rindE3 / rindE1)
    rindNu32 = rindNu23 * (rindE3 / rindE2)
    rindNu21 = rindNu12 * (rindE2 / rindE1)
    cons = [{'type':'ineq','fun':pithE1 - (pithE1_avg + 2*pithE1_mu)},
            {'type':'ineq','fun':(pithE1_avg - 2*pithE1_mu) - pithE1},
            {'type':'ineq','fun':pithE2 - (pithE1_avg + 2*pithE1_mu)},
            {'type':'ineq','fun':(pithE1_avg - 2*pithE1_mu) - pithE2},
            {'type':'ineq','fun':pithE3 - pithE3_high},
            {'type':'ineq','fun':pithE3_low - pithE3},
            {'type':'ineq','fun':pithNu12 - pithNu12_high},
            {'type':'ineq','fun':pithNu12_low - pithNu12},
            {'type':'ineq','fun':pithNu13 - pithNu13_high},
            {'type':'ineq','fun':pithNu13_low - pithNu13},
            {'type':'ineq','fun':pithNu23 - pithNu13_high},
            {'type':'ineq','fun':pithNu13_low - pithNu23},
            {'type':'eq','fun':pithG12 - (pithE1/(2*(1+pithNu12)))},
            {'type':'ineq','fun':pithG13 - pithG13_high_constant*pithE3},
            {'type':'ineq','fun':pithG13_low_constant*pithE3 - pithG13},
            {'type':'ineq','fun':pithG23 - pithG13_high_constant*pithE3},
            {'type':'ineq','fun':pithG13_low_constant*pithE3 - pithG23},
            {'type':'ineq','fun':1-pithNu12*pithNu21-pithNu23*pithNu32-pithNu13*pithNu31-2*pithNu21*pithNu32*pithNu13},
            {'type':'ineq','fun':rindE1 - (rindE1_avg + 2*rindE1_mu)},
            {'type':'ineq','fun':(rindE1_avg - 2*rindE1_mu) - rindE1},
            {'type':'ineq','fun':rindE2 - (rindE1_avg + 2*rindE1_mu)},
            {'type':'ineq','fun':(rindE1_avg - 2*rindE1_mu) - rindE2},
            {'type':'ineq','fun':rindNu12 - rindNu12_high},
            {'type':'ineq','fun':rindNu12_low - rindNu12},
            {'type':'ineq','fun':rindNu13 - rindNu13_high},
            {'type':'ineq','fun':rindNu13_low - rindNu13},
            {'type':'ineq','fun':rindNu23 - rindNu13_high},
            {'type':'ineq','fun':rindNu13_low - rindNu23},
            {'type':'eq','fun':rindG12 - (rindE1/(2*(1+rindNu12)))},
            {'type':'ineq','fun':rindG13 - rindG13_high_constant*rindE3},
            {'type':'ineq','fun':rindG13_low_constant*rindE3 - rindG13},
            {'type':'ineq','fun':rindG23 - rindG13_high_constant*rindE3},
            {'type':'ineq','fun':rindG13_low_constant*rindE3 - rindG23},
            {'type':'ineq','fun':1-rindNu12*rindNu21-rindNu23*rindNu32-rindNu13*rindNu31-2*rindNu21*rindNu32*rindNu13},
            ]
    return cons

init0 = np.array([pithE1,pithE2,pithE3,pithNu12,pithNu13,pithNu23,pithG12,pithG13,pithG23,rindE1,rindE2,rindE3,rindNu12,rindNu13,rindNu23,rindG12,rindG13,rindG23])

cons = constraints(init0)
print(cons)
res = minimize(fun=ObjectiveFunction,x0=init0,constraints=constraints(init0),callback=callb,options={'maxiter':10})

