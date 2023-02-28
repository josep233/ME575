def con1(init):
       rindG12 = 0.3
       rindE1 = 850
       rindNu12 = 0.3
       return rindG12 - (rindE1/(2*(1+rindNu12)))
def con2(init):
       rindG13 = init[16]
       rindE3 = init[11]
       return 0.21*rindE3 - rindG13
def con3(init):
       rindG13 = init[16]
       rindE3 = init[11]
       return rindG13 - 0.03*rindE3
def con4(init):
       rindG23 = init[17]
       rindE3 = init[11]
       return 0.21*rindE3 - rindG23
def con5(init):
       rindG23 = init[17]
       rindE3 = init[11]
       return rindG23 - 0.03*rindE3

def con6(out):
       return 40 - out

def constraints(init,out):
    cons = [{'type':'ineq','fun':con1},
            {'type':'ineq','fun':con2},
            {'type':'ineq','fun':con3},
            {'type':'ineq','fun':con4},
            {'type':'ineq','fun':con5},
            {'type':'ineq','fun':con6},
            ]
    return cons



