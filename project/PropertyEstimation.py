#PURPOSE: the purpose of this file is to hold the material property range values for each of the 18 material properties used in a model.

import RindE3 as r
import numpy as np
import math

def properties(stalk):
    rindG13_low_constant = 0.03
    rindG13_high_constant = 0.21
    rindE1_avg = 0.85 * 1000
    rindNu12_low = 0.2
    rindNu12_high = 0.45
    rindNu13_low = 0.009
    rindNu13_high = 0.086
    pithG13_low_constant = 0.03
    pithG13_high_constant = 0.21
    pithE1_avg = 0.026 * 1000
    pithNu12_low = 0.2
    pithNu12_high = 0.45
    pithNu13_low = 0.009
    pithNu13_high = 0.086
    pithE3_low = 0.06 * 1000
    pithE3_high = 0.18 * 1000
    rindE3 = r.rinde3(stalk) * 1000

    pithE1=pithE2=pithE3=pithNu12=pithNu13=pithNu23=pithG12=pithG13=pithG23=rindE1=rindE2=rindNu12=rindNu13=rindNu23=rindG12=rindG13=rindG23=rindNu31=pithNu31=1
    
    while rindNu12**2 + 2*rindNu13*rindNu31 + 2*rindNu12*rindNu13*rindNu31 > 1:
        rindG13 = np.mean([rindG13_low_constant,rindG13_high_constant]) * rindE3
        rindG23 = rindG13
        rindE1 = rindE1_avg
        rindE2 = rindE1
        rindNu12 = np.mean([rindNu12_low,rindNu12_high])
        rindG12 = rindE1 / (2*(1+rindNu12))
        rindNu13 = np.mean([rindNu13_low,rindNu13_high])
        rindNu23 = rindNu13
        rindNu31 = rindNu13 * (rindE3 / rindE1)
    pithNu12 = rindNu12
    rindprops = (rindE1,rindE2,rindE3,rindNu12,rindNu13,rindNu23,rindG12,rindG13,rindG23,)
    while pithNu12**2 + 2*pithNu13*pithNu31 + 2*pithNu12*pithNu13*pithNu31 > 1:
        pithE3 = np.mean([pithE3_low,pithE3_high])
        pithG13 = np.mean([pithG13_low_constant,pithG13_high_constant]) * pithE3
        pithG23 = pithG13
        pithE1 = pithE1_avg
        pithE2 = pithE1
        pithNu12 = np.mean([pithNu12_low,pithNu12_high])
        pithG12 = pithE1 / (2*(1+pithNu12))
        pithNu13 = np.mean([pithNu13_low,pithNu13_high])
        pithNu23 = pithNu13
        pithNu31 = pithNu13 * (pithE3 / pithE1)
    pithprops = (pithE1,pithE2,pithE3,pithNu12,pithNu13,pithNu23,pithG12,pithG13,pithG23,)
    return pithprops,rindprops