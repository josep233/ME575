import PropertyEstimation as p
import ABQScript as a
import numpy as np
import os

def ObjectiveFunction(stalk,props):
    a.script(stalk,props)
    os.system('abaqus python C:\\Users\\Joseph\Desktop\\Temp\\test.py')

