import RindE3 as r
import ABQScript as a
import numpy as np
import subprocess

def ObjectiveFunction(init):
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
    rindG12 = init[0]
    rindG13 = init[1]
    rindG23 = init[2]
    
    pithprops = (pithE1,pithE2,pithE3,pithNu12,pithNu13,pithNu23,pithG12,pithG13,pithG23,)
    rindprops = (rindE1,rindE2,rindE3,rindNu12,rindNu13,rindNu23,rindG12,rindG13,rindG23,)
    a.script(pithprops,rindprops)
    subprocess.run("abaqus cae noGUI=C:\\Users\\Joseph\\Desktop\\Temp\\test.py", shell=True)
    file = open('C:\\Users\\Joseph\\Desktop\\Temp\\test.txt',"r")
    ans = np.double(file.readline())
    ans = float(ans)
    print("function evaluation: ",Nfeval)
    # print("pithE1: ",init[0],"pithE2: ",init[1],"pithE3: ",init[2],"pithNu12: ",init[3],"pithNu13: ",init[4],"pithNu23: ",init[5],"pithG12: ",init[6],"pithG13: ",init[7],"pithG23: ",init[8],"rindE1: ",init[9],"rindE2: ",init[10],"rindE3: ",init[11],"rindNu12: ",init[12],"rindNu13: ",init[13],"rindNu23: ",init[14],"rindG12: ",init[15],"rindG13: ",init[16],"rindG23: ",init[17])
    print("eigenvalue: ",ans)
    Nfeval += 1
    return ans

