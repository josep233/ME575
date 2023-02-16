import numpy as np

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

def con1(init,pithE1_avg,pithE1_mu):
       pithE1 = init[0]
       return pithE1 - (pithE1_avg + 2*pithE1_mu)
def con2(init,pithE1_avg,pithE1_mu):
       pithE1 = init[0]
       return (pithE1_avg - 2*pithE1_mu) - pithE1
def con3(init,pithE1_avg,pithE1_mu):
       pithE2 = init[1]
       return pithE2 - (pithE1_avg + 2*pithE1_mu)
def con4(init,pithE1_avg,pithE1_mu):
       pithE2 = init[1]
       return (pithE1_avg - 2*pithE1_mu) - pithE2
def con5(init,pithE3_high):
       pithE3 = init[2]
       return pithE3 - pithE3_high
def con6(init,pithE3_low):
       pithE3 = init[2]
       return pithE3_low - pithE3
def con7(init,pithNu12_high):
       pithNu12 = init[3]
       return pithNu12 - pithNu12_high
def con8(init,pithNu12_low):
       pithNu12 = init[3]
       return pithNu12_low - pithNu12
def con9(init,pithNu13_high):
       pithNu13 = init[4]
       return pithNu13 - pithNu13_high
def con10(init,pithNu13_low):
       pithNu13 = init[4]
       return pithNu13_low - pithNu13
def con11(init,pithNu13_high):
       pithNu23 = init[5]
       return pithNu23 - pithNu13_high
def con12(init,pithNu13_low):
       pithNu23 = init[5]
       return pithNu13_low - pithNu23
def con13(init):
       pithG12 = init[6]
       pithE1 = init[0]
       pithNu23 = init[3]
       return pithG12 - (pithE1/(2*(1+pithNu23)))
def con14(init,pithG13_high_constant):
       pithG13 = init[6]
       return init[4] - pithG13_high_constant*init[2]
def con15(init,pithG13_low_constant):
       return pithG13_low_constant*init[2] - init[9]
def con16(init,pithG13_high_constant):
       return init[10] - pithG13_high_constant*init[2]
def con17(init,pithG13_low_constant):
       return pithG13_low_constant*init[2] - init[10]
def con18(init,pithNu13_low):
       return pithNu13_low - init[5]



def constraints(init):
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
            {'type':'eq','fun':con13},
            {'type':'ineq','fun':con14},
            {'type':'ineq','fun':con15},
            {'type':'ineq','fun':con16},
            {'type':'ineq','fun':con17},
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



