import PropertyEstimation as p

def script(stalk,pithprops,rindprops):
    txt = f"""
mdb = Mdb()
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
from odbAccess import *
from odbMaterial import *
from odbSection import *
from abaqusConstants import *
import regionToolset
def run():
    n = 1
    if n == 1:
        openMdb("C:\\Users\\Joseph\\Desktop\\Personal\\ME575\\project\\Proj\\ProjCAE\\Stalk_{stalk}-quarter_buckling-Folder-1")
        mdb.models['Model-1'].Material('rind')
        mdb.models['Model-1'].Material('pith')
        buckling = open(r'C:\\Users\\Joseph\Desktop\\Temp\\test.txt','w')
        
        mdb.models['Model-1'].materials['pith'].Elastic(table=({pithprops}, ), type=ENGINEERING_CONSTANTS)
        mdb.models['Model-1'].materials['rind'].Elastic(table=({rindprops}, ), type=ENGINEERING_CONSTANTS)

        jobname = 'test{stalk}'
        mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF,explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=ON,memory=90, memoryUnits=PERCENTAGE, model='Model-1', modelPrint=OFF,multiprocessingMode=DEFAULT, name=jobname, nodalOutputPrecision=SINGLE,numCpus=1, numGPUs=0, queue=None, resultsFormat=ODB, scratch='', type=ANALYSIS, userSubroutine='', waitHours=0, waitMinutes=0)
        myJob=mdb.jobs[jobname]
        myJob.submit(consistencyChecking=OFF)
        myJob.waitForCompletion()
        
        odb = openOdb(path=jobname+'.odb')
        
        mode = odb.steps['BuckleStep'].frames[1].description
        dat = str(mode)
        ans = dat[-6:]
        buckling.write(ans)
        buckling.close()
    return ans
ans = run()
"""
    location = "C:\\Users\\Joseph\Desktop\\Temp\\test.py"
    file = open(location,'w')
    file.write(txt)
    file.close()