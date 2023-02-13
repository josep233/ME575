import PropertyEstimation as p
import ABQScript as a
import numpy as np
import subprocess

def ObjectiveFunction(stalk):
    pithprops = p.properties(stalk)[0]
    rindprops = p.properties(stalk)[1]
    a.script(stalk,pithprops,rindprops)
    ans = subprocess.Popen("abaqus cae noGUI=C:\\Users\\Joseph\\Desktop\\Temp\\test.py",stdout=subprocess.PIPE,shell=True)
    print(ans.communicate())

 
stalk = 7
ObjectiveFunction(stalk)
