import PropertyEstimation as p
import ABQScript as a
import numpy as np
import subprocess

def ObjectiveFunction(stalk):
    pithprops = p.properties(stalk)[0]
    rindprops = p.properties(stalk)[1]
    a.script(stalk,pithprops,rindprops)
    subprocess.run("abaqus cae noGUI=C:\\Users\\Joseph\\Desktop\\Temp\\test.py", shell=True)
    file = open('C:\\Users\\Joseph\\Desktop\\Temp\\test.txt',"r")
    ans = file.readline()
    return ans


 
stalk = 7
ans = ObjectiveFunction(stalk)
print(ans)

