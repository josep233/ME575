import numpy as np
import RindE3 as r

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

def con28(init):
       rindG12 = 0.3
       rindE1 = 850
       rindNu12 = 0.3
       return rindG12 - (rindE1/(2*(1+rindNu12)))
def con29(init):
       rindG13 = init[16]
       rindE3 = init[11]
       return rindG13_high_constant*rindE3 - rindG13
def con30(init):
       rindG13 = init[16]
       rindE3 = init[11]
       return rindG13 - rindG13_low_constant*rindE3
def con31(init):
       rindG23 = init[17]
       rindE3 = init[11]
       return rindG13_high_constant*rindE3 - rindG23
def con32(init):
       rindG23 = init[17]
       rindE3 = init[11]
       return rindG23 - rindG13_low_constant*rindE3

def con35(out):
       return 14 - out

def con36(init,stalk):
       rindE3 = init[11]
       return rindE3 - r.rinde3(stalk)

def constraints(init,out,stalk):
    cons = [{'type':'ineq','fun':con1},
            {'type':'ineq','fun':con2},
            {'type':'ineq','fun':con3},
            {'type':'ineq','fun':con4},
            {'type':'ineq','fun':con5},
            {'type':'ineq','fun':con6},
            {'type':'ineq','fun':con7},
            {'type':'ineq','fun':con8},
            {'type':'ineq','fun':con9},
            {'type':'ineq','fun':con10},
            {'type':'ineq','fun':con11},
            {'type':'ineq','fun':con12},
       #      {'type':'eq','fun':con13},
            {'type':'ineq','fun':con14},
            {'type':'ineq','fun':con15},
            {'type':'ineq','fun':con16},
            {'type':'ineq','fun':con17},
            {'type':'ineq','fun':con18},
            {'type':'ineq','fun':con19},
            {'type':'ineq','fun':con20},
            {'type':'ineq','fun':con21},
            {'type':'ineq','fun':con22},
            {'type':'ineq','fun':con23},
            {'type':'ineq','fun':con24},
            {'type':'ineq','fun':con25},
            {'type':'ineq','fun':con26},
            {'type':'ineq','fun':con27},
       #      {'type':'eq','fun':con28},
            {'type':'ineq','fun':con29},
            {'type':'ineq','fun':con30},
            {'type':'ineq','fun':con31},
            {'type':'ineq','fun':con32},
            {'type':'ineq','fun':con33},
            {'type':'ineq','fun':con34},
            {'type':'eq','fun':con35},
            {'type':'eq','fun':con36,'args':(stalk,)},
            ]
    return cons



