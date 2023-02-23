# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2022.HF4 replay file
# Internal Version: 2022_07_25-12.30.49 176790
# Run by Joseph on Wed Feb 22 23:17:48 2023
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(1.32292, 1.32407), width=194.733, 
    height=131.348)
session.viewports['Viewport: 1'].makeCurrent()
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
execfile('C:/Users/Joseph/Desktop/Temp/test.py', __main__.__dict__)
#: A new model database has been created.
#: The model "Model-1" has been created.
session.viewports['Viewport: 1'].setValues(displayedObject=None)
#: The model database "C:\Users\Joseph\Desktop\Personal\ME575\project\Proj\ProjCAE\Stalk_7-quarter_buckling-Folder-1.cae" has been opened.
session.viewports['Viewport: 1'].setValues(displayedObject=None)
#: Model: C:/Users/Joseph/Desktop/Personal/ME575/test7.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     2
#: Number of Meshes:             3
#: Number of Element Sets:       10
#: Number of Node Sets:          23
#: Number of Steps:              1
#* IndexError: Sequence index out of range
#* File "C:/Users/Joseph/Desktop/Temp/test.py", line 47, in <module>
#*     ans = run()
#* File "C:/Users/Joseph/Desktop/Temp/test.py", line 40, in run
#*     mode = odb.steps['BuckleStep'].frames[1].description
