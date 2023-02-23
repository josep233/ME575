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

# pithNu31 = pithNu13 * (pithE3 / pithE1)
# pithNu32 = pithNu23 * (pithE3 / pithE2)
# pithNu21 = pithNu12 * (pithE2 / pithE1)
# rindNu31 = rindNu13 * (rindE3 / rindE1)
# rindNu32 = rindNu23 * (rindE3 / rindE2)
# rindNu21 = rindNu12 * (rindE2 / rindE1)

def con1(init):
       pithE1 = init[0]
       return pithE1 - (pithE1_avg - 2*pithE1_mu)
def con2(init):
       pithE1 = init[0]
       return (pithE1_avg + 2*pithE1_mu) - pithE1
def con3(init):
       pithE2 = init[1]
       return pithE2 - (pithE1_avg - 2*pithE1_mu)
def con4(init):
       pithE2 = init[1]
       return (pithE1_avg + 2*pithE1_mu) - pithE2
def con5(init):
       pithE3 = init[2]
       return pithE3_high - pithE3
def con6(init):
       pithE3 = init[2]
       return pithE3 - pithE3_low
def con7(init):
       pithNu12 = init[3]
       return pithNu12_high -pithNu12
def con8(init):
       pithNu12 = init[3]
       return pithNu12 -pithNu12_low
def con9(init):
       pithNu13 = init[4]
       return pithNu13_high - pithNu13
def con10(init):
       pithNu13 = init[4]
       return pithNu13 - pithNu13_low
def con11(init):
       pithNu23 = init[5]
       return pithNu13_high - pithNu23
def con12(init):
       pithNu23 = init[5]
       return pithNu23 - pithNu13_low
def con13(init):
       pithG12 = init[6]
       pithE1 = init[0]
       pithNu23 = init[3]
       return pithG12 - (pithE1/(2*(1+pithNu23)))
def con14(init):
       pithG13 = init[6]
       pithE3 = init[2]
       return pithG13_high_constant*pithE3 - pithG13
def con15(init):
       pithG13 = init[6]
       pithE3 = init[2]
       return pithG13 - pithG13_low_constant*pithE3
def con16(init):
       pithG23 = init[7]
       pithE3 = init[2]
       return pithG13_high_constant*pithE3 - pithG23
def con17(init):
       pithG23 = init[7]
       pithE3 = init[2]
       return pithG23 - pithG13_low_constant*pithE3

def con18(init):
       rindE1 = init[0]
       return rindE1 - (rindE1_avg - 2*rindE1_mu)
def con19(init):
       rindE1 = init[0]
       return (rindE1_avg + 2*rindE1_mu) - rindE1
def con20(init):
       rindE2 = init[1]
       return rindE2 - (rindE1_avg - 2*rindE1_mu)
def con21(init):
       rindE2 = init[1]
       return (rindE1_avg + 2*rindE1_mu) - rindE2
def con22(init):
       rindNu12 = init[3]
       return rindNu12_high - rindNu12
def con23(init):
       rindNu12 = init[3]
       return rindNu12 - rindNu12_low
def con24(init):
       rindNu13 = init[4]
       return rindNu13_high - rindNu13
def con25(init):
       rindNu13 = init[4]
       return rindNu13 - rindNu13_low
def con26(init):
       rindNu23 = init[5]
       return rindNu13_high - rindNu23
def con27(init):
       rindNu23 = init[5]
       return rindNu23 - rindNu13_low
def con28(init):
       rindG12 = init[6]
       rindE1 = init[0]
       rindNu23 = init[3]
       return rindG12 - (rindE1/(2*(1+rindNu23)))
def con29(init):
       rindG13 = init[6]
       rindE3 = init[2]
       return rindG13_high_constant*rindE3 - rindG13
def con30(init):
       rindG13 = init[6]
       rindE3 = init[2]
       return rindG13 - rindG13_low_constant*rindE3
def con31(init):
       rindG23 = init[7]
       rindE3 = init[2]
       return rindG13_high_constant*rindE3 - rindG23
def con32(init):
       rindG23 = init[7]
       rindE3 = init[2]
       return rindG23 - rindG13_low_constant*rindE3

def con33(init):
       pithNu12 = init[3]
       pithNu13 = init[4]
       pithNu23 = init[5]
       pithE3 = init[2]
       pithE1 = init[0]
       pithE2 = init[1]
       pithNu31 = pithNu13 * (pithE3 / pithE1)
       pithNu32 = pithNu23 * (pithE3 / pithE2)
       pithNu21 = pithNu12 * (pithE2 / pithE1)
       return 1 - pithNu12*pithNu21 - pithNu23*pithNu32 - pithNu31*pithNu13 - 2*pithNu21*pithNu32*pithNu13

def con34(init):
       rindNu12 = init[12]
       rindNu13 = init[13]
       rindNu23 = init[14]
       rindE3 = init[11]
       rindE1 = init[9]
       rindE2 = init[10]
       rindNu31 = rindNu13 * (rindE3 / rindE1)
       rindNu32 = rindNu23 * (rindE3 / rindE2)
       rindNu21 = rindNu12 * (rindE2 / rindE1)
       return 1 - rindNu12*rindNu21 - rindNu23*rindNu32 - rindNu31*rindNu13 - 2*rindNu21*rindNu32*rindNu13

def con35(out):
       return 14 - out

def con36(init,stalk):
       rindE3 = init[11]
       return rindE3 - r.rinde3(stalk)

def constraints(init,out,stalk):
    cons = [{'type':'ineq','fun':con1},
       #      {'type':'ineq','fun':con2},
            {'type':'ineq','fun':con3},
       #      {'type':'ineq','fun':con4},
            {'type':'ineq','fun':con5},
       #      {'type':'ineq','fun':con6},
            {'type':'ineq','fun':con7},
       #      {'type':'ineq','fun':con8},
            {'type':'ineq','fun':con9},
       #      {'type':'ineq','fun':con10},
            {'type':'ineq','fun':con11},
       #      {'type':'ineq','fun':con12},
       #      {'type':'eq','fun':con13},
            {'type':'ineq','fun':con14},
       #      {'type':'ineq','fun':con15},
            {'type':'ineq','fun':con16},
       #      {'type':'ineq','fun':con17},
            {'type':'ineq','fun':con18},
       #      {'type':'ineq','fun':con19},
            {'type':'ineq','fun':con20},
       #      {'type':'ineq','fun':con21},
            {'type':'ineq','fun':con22},
       #      {'type':'ineq','fun':con23},
            {'type':'ineq','fun':con24},
       #      {'type':'ineq','fun':con25},
            {'type':'ineq','fun':con26},
       #      {'type':'ineq','fun':con27},
       #      {'type':'eq','fun':con28},
            {'type':'ineq','fun':con29},
       #      {'type':'ineq','fun':con30},
            {'type':'ineq','fun':con31},
       #      {'type':'ineq','fun':con32},
            {'type':'ineq','fun':con33},
            {'type':'ineq','fun':con34},
            {'type':'ineq','fun':con35},
            {'type':'eq','fun':con36,'args':(stalk,)},
            ]
    return cons



