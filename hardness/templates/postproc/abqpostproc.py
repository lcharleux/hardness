# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import __main__
from argiope.abq.abqpostproc import write_xy_report, write_field_report

odbPath     = "#ODBPATH"
histPath    = "#HISTPATH"
contactPath = "#CONTACTPATH"
fieldPath   = "#FIELDPATH"

o1 = session.openOdb(name = odbPath)
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
odb = session.odbs[odbPath]

# CONTACT DATA
surface_nodes = [n.label for n in  odb.rootAssembly.instances["I_SAMPLE"].nodeSets["SURFACE"].nodes]
tags =  ["Coordinates: COOR1 PI: I_SAMPLE Node {0} in NSET SURFACE".format(l) for l in surface_nodes]
tags += ["Coordinates: COOR2 PI: I_SAMPLE Node {0} in NSET SURFACE".format(l) for l in surface_nodes]
tags += ["Contact pressure: CPRESS   ASSEMBLY_I_SAMPLE_SURFACE/ASSEMBLY_I_INDENTER_SURFACE_FACES PI: I_SAMPLE Node {0}".format(l) for l in surface_nodes]
cols =  ["COOR1_n{0}".format(l) for l in surface_nodes]
cols += ["COOR2_n{0}".format(l) for l in surface_nodes]
cols += ["CPRESS_n{0}".format(l) for l in surface_nodes]

write_xy_report(odb, contactPath + ".rpt", 
    tags = tags,
    columns = cols,
    steps   = ('LOADING1', 'LOADING2'))

# HISTORY OUTPUTS
session.XYDataFromHistory(name='Wtot', odb=odb, 
    outputVariableName='External work: ALLWK for Whole Model', 
    steps=('LOADING1', 'LOADING2', ),  )
session.XYDataFromHistory(name='Wf', odb=odb, 
    outputVariableName='Frictional dissipation: ALLFD for Whole Model', 
    steps=('LOADING1', 'LOADING2', ),  )
session.XYDataFromHistory(name='Wps', odb=odb, 
    outputVariableName='Plastic dissipation: ALLPD PI: I_SAMPLE in ELSET ALL_ELEMENTS', 
    steps=('LOADING1', 'LOADING2', ),  )
session.XYDataFromHistory(name='Wei', odb=odb, 
    outputVariableName='Strain energy: ALLSE PI: I_INDENTER in ELSET ALL_ELEMENTS', 
    steps=('LOADING1', 'LOADING2', ),  )
session.XYDataFromHistory(name='Wes', odb=odb, 
    outputVariableName='Strain energy: ALLSE PI: I_SAMPLE in ELSET ALL_ELEMENTS', 
    steps=('LOADING1', 'LOADING2', ),  )
session.XYDataFromHistory(name='F', odb=odb, 
    outputVariableName='Reaction force: RF2 PI: I_INDENTER Node 296 in NSET REF_NODE', 
    steps=('LOADING1', 'LOADING2', ), )
session.XYDataFromHistory(name='ht', odb=odb, 
    outputVariableName='Spatial displacement: U2 PI: I_INDENTER Node 73 in NSET TIP_NODE', 
    steps=('LOADING1', 'LOADING2', ),  )
session.XYDataFromHistory(name='h', odb=odb, 
    outputVariableName='Spatial displacement: U2 PI: I_INDENTER Node 296 in NSET REF_NODE', 
    steps=('LOADING1', 'LOADING2', ),  )

x0 = session.xyDataObjects['Wf']
x1 = session.xyDataObjects['Wps']
x2 = session.xyDataObjects['Wei']
x3 = session.xyDataObjects['Wes']
x4 = session.xyDataObjects['F']
x5 = session.xyDataObjects['ht']
x6 = session.xyDataObjects['h']
session.writeXYReport(fileName=histPath + ".rpt", 
                      xyData=(x0, x1, x2, x3, x4, x5, x6),
                      append = False)


# FIELD OUTPUTS
variable = (('S', INTEGRATION_POINT, 
                  ((COMPONENT, 'S11'),  
                   (COMPONENT, 'S22'), 
                   (COMPONENT, 'S33'), 
                   (COMPONENT, 'S12'), 
                  )), )

write_field_report(odb, 
                   path = fieldPath + ".rpt", 
                   variable = variable,
                   instance = 'I_SAMPLE', 
                   output_position = NODAL, 
                   step = -1 , 
                   frame = -1)

