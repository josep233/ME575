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

def script(stalk,props):
    n = 1
    if n == 1:
        openMdb("C:\Users\Joseph\Box\CropBiomechanics\COMPUTATIONALMODELING\Inputs\VarSensInit\VarSensInitCAE\Stalk_"+stalk)
        mdb.models['Model-1'].Material('rind')
        mdb.models['Model-1'].Material('pith')
        
        mdb.models['Model-1'].materials['pith'].Elastic(table=(props, ), type=ENGINEERING_CONSTANTS)
        mdb.models['Model-1'].materials['rind'].Elastic(table=(props, ), type=ENGINEERING_CONSTANTS)
