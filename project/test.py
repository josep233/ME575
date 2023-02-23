import Constraints as c
import ObjectiveFunction as f
import RindE3 as r
import numpy as np
from scipy.optimize import minimize
import subprocess
import ABQScript as a

const = 0
stalk = 7
pithE1 = 0.026*10**(-const)
pithE2 = pithE1
pithE3 = 0.11*10**(-const)
pithNu12 = 0.3*10**(-const)
pithNu13 = 0.05*10**(-const)
pithNu23 = pithNu13
pithG12 = pithE1 / (2*(1+pithNu12))
pithG13 = 0.1 * pithE3
pithG23 = pithG13
rindE1 = 0.85*10**(-const)
rindE2 = rindE1
rindE3 = r.rinde3(stalk)*10**(-const)
rindNu12 = pithNu12
rindNu13 = pithNu13
rindNu23 = pithNu23
rindG12 = rindE1 / (2*(1+rindNu12))
rindG13 = 0.1 * rindE3
rindG23 = rindG13


Nfeval = 1
def callb():
    global Nfeval
    print("hello?")
    print(Nfeval)
    print(ObjectiveFunction(init0))
    Nfeval += 1

def ObjectiveFunction(init):
    global Nfeval
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
    
    pithprops = (pithE1 * 1000*10**(const),pithE2 * 1000*10**(const),pithE3 * 1000*10**(const),pithNu12*1*10**(const),pithNu13*1*10**(const),pithNu23*1*10**(const),pithG12 * 1000*10**(const),pithG13 * 1000*10**(const),pithG23 * 1000*10**(const),)
    rindprops = (rindE1 * 1000*10**(const),rindE2 * 1000*10**(const),rindE3 * 1000*10**(const),rindNu12*1*10**(const),rindNu13*1*10**(const),rindNu23*1*10**(const),rindG12 * 1000*10**(const),rindG13 * 1000*10**(const),rindG23 * 1000*10**(const),)
    a.script(stalk,pithprops,rindprops)
    subprocess.run("abaqus cae noGUI=C:\\Users\\Joseph\\Desktop\\Temp\\test.py", shell=True)
    file = open('C:\\Users\\Joseph\\Desktop\\Temp\\test.txt',"r")
    ans = file.readline()
    ans = float(ans)
    print("function evaluation: ",Nfeval)
    # print("pithE1: ",init[0],"pithE2: ",init[1],"pithE3: ",init[2],"pithNu12: ",init[3],"pithNu13: ",init[4],"pithNu23: ",init[5],"pithG12: ",init[6],"pithG13: ",init[7],"pithG23: ",init[8],"rindE1: ",init[9],"rindE2: ",init[10],"rindE3: ",init[11],"rindNu12: ",init[12],"rindNu13: ",init[13],"rindNu23: ",init[14],"rindG12: ",init[15],"rindG13: ",init[16],"rindG23: ",init[17])
    print("eigenvalue: ",ans)
    Nfeval += 1
    return ans

init0 = np.array([pithE1,pithE2,pithE3,pithNu12,pithNu13,pithNu23,pithG12,pithG13,pithG23,rindE1,rindE2,rindE3,rindNu12,rindNu13,rindNu23,rindG12,rindG13,rindG23])
out = ObjectiveFunction
res = minimize(fun=ObjectiveFunction,x0=init0,constraints=c.constraints(init0,out,stalk))
print(res)

