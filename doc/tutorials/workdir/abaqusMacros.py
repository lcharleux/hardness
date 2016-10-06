# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__

def Macro1():
    import visualization
    import xyPlot
    import displayGroupOdbToolset as dgo
    odb = session.odbs['/home/lcharleux/Documents/Programmation/Python/Modules/hardness/doc/tutorials/workdir/indentation_2D.odb']
    session.XYDataFromHistory(name='XYData-1', odb=odb, 
        outputVariableName='Contact pressure: CPRESS   ASSEMBLY_I_SAMPLE_SURFACE/ASSEMBLY_I_INDENTER_SURFACE PI: I_SAMPLE Node 67', 
        steps=('LOADING1', 'LOADING2', ), )
    session.XYDataFromHistory(name='XYData-2', odb=odb, 
        outputVariableName='Contact pressure: CPRESS   ASSEMBLY_I_SAMPLE_SURFACE/ASSEMBLY_I_INDENTER_SURFACE PI: I_SAMPLE Node 68', 
        steps=('LOADING1', 'LOADING2', ), )
    session.XYDataFromHistory(name='XYData-3', odb=odb, 
        outputVariableName='Contact pressure: CPRESS   ASSEMBLY_I_SAMPLE_SURFACE/ASSEMBLY_I_INDENTER_SURFACE PI: I_SAMPLE Node 69', 
        steps=('LOADING1', 'LOADING2', ), )
    session.XYDataFromHistory(name='XYData-4', odb=odb, 
        outputVariableName='Contact pressure: CPRESS   ASSEMBLY_I_SAMPLE_SURFACE/ASSEMBLY_I_INDENTER_SURFACE PI: I_SAMPLE Node 70', 
        steps=('LOADING1', 'LOADING2', ), )
    session.XYDataFromHistory(name='XYData-5', odb=odb, 
        outputVariableName='Contact pressure: CPRESS   ASSEMBLY_I_SAMPLE_SURFACE/ASSEMBLY_I_INDENTER_SURFACE PI: I_SAMPLE Node 71', 
        steps=('LOADING1', 'LOADING2', ), )
    session.XYDataFromHistory(name='XYData-6', odb=odb, 
        outputVariableName='Contact pressure: CPRESS   ASSEMBLY_I_SAMPLE_SURFACE/ASSEMBLY_I_INDENTER_SURFACE PI: I_SAMPLE Node 72', 
        steps=('LOADING1', 'LOADING2', ), )
    session.XYDataFromHistory(name='XYData-7', odb=odb, 
        outputVariableName='Contact pressure: CPRESS   ASSEMBLY_I_SAMPLE_SURFACE/ASSEMBLY_I_INDENTER_SURFACE PI: I_SAMPLE Node 73', 
        steps=('LOADING1', 'LOADING2', ), )
    session.XYDataFromHistory(name='XYData-8', odb=odb, 
        outputVariableName='Contact pressure: CPRESS   ASSEMBLY_I_SAMPLE_SURFACE/ASSEMBLY_I_INDENTER_SURFACE PI: I_SAMPLE Node 74', 
        steps=('LOADING1', 'LOADING2', ), )
    odb = session.odbs['/home/lcharleux/Documents/Programmation/Python/Modules/hardness/doc/tutorials/workdir/indentation_2D.odb']
    session.XYDataFromHistory(name='XYData-9', odb=odb, 
        outputVariableName='Contact pressure: CPRESS   ASSEMBLY_I_SAMPLE_SURFACE/ASSEMBLY_I_INDENTER_SURFACE PI: I_SAMPLE Node 67', 
        steps=('LOADING1', 'LOADING2', ), )
    session.XYDataFromHistory(name='XYData-10', odb=odb, 
        outputVariableName='Contact pressure: CPRESS   ASSEMBLY_I_SAMPLE_SURFACE/ASSEMBLY_I_INDENTER_SURFACE PI: I_SAMPLE Node 68', 
        steps=('LOADING1', 'LOADING2', ), )
    session.XYDataFromHistory(name='XYData-11', odb=odb, 
        outputVariableName='Contact pressure: CPRESS   ASSEMBLY_I_SAMPLE_SURFACE/ASSEMBLY_I_INDENTER_SURFACE PI: I_SAMPLE Node 69', 
        steps=('LOADING1', 'LOADING2', ), )
    session.XYDataFromHistory(name='XYData-12', odb=odb, 
        outputVariableName='Contact pressure: CPRESS   ASSEMBLY_I_SAMPLE_SURFACE/ASSEMBLY_I_INDENTER_SURFACE PI: I_SAMPLE Node 70', 
        steps=('LOADING1', 'LOADING2', ), )
    session.XYDataFromHistory(name='XYData-13', odb=odb, 
        outputVariableName='Contact pressure: CPRESS   ASSEMBLY_I_SAMPLE_SURFACE/ASSEMBLY_I_INDENTER_SURFACE PI: I_SAMPLE Node 71', 
        steps=('LOADING1', 'LOADING2', ), )
    session.XYDataFromHistory(name='XYData-14', odb=odb, 
        outputVariableName='Contact pressure: CPRESS   ASSEMBLY_I_SAMPLE_SURFACE/ASSEMBLY_I_INDENTER_SURFACE PI: I_SAMPLE Node 72', 
        steps=('LOADING1', 'LOADING2', ), )
    session.XYDataFromHistory(name='XYData-15', odb=odb, 
        outputVariableName='Contact pressure: CPRESS   ASSEMBLY_I_SAMPLE_SURFACE/ASSEMBLY_I_INDENTER_SURFACE PI: I_SAMPLE Node 73', 
        steps=('LOADING1', 'LOADING2', ), )
    session.XYDataFromHistory(name='XYData-16', odb=odb, 
        outputVariableName='Contact pressure: CPRESS   ASSEMBLY_I_SAMPLE_SURFACE/ASSEMBLY_I_INDENTER_SURFACE PI: I_SAMPLE Node 74', 
        steps=('LOADING1', 'LOADING2', ), )
    x0 = session.xyDataObjects['XYData-1']
    x1 = session.xyDataObjects['XYData-2']
    x2 = session.xyDataObjects['XYData-3']
    x3 = session.xyDataObjects['XYData-4']
    x4 = session.xyDataObjects['XYData-5']
    x5 = session.xyDataObjects['XYData-6']
    x6 = session.xyDataObjects['XYData-7']
    x7 = session.xyDataObjects['XYData-8']
    x8 = session.xyDataObjects['XYData-9']
    x9 = session.xyDataObjects['XYData-10']
    x10 = session.xyDataObjects['XYData-11']
    x11 = session.xyDataObjects['XYData-12']
    x12 = session.xyDataObjects['XYData-13']
    x13 = session.xyDataObjects['XYData-14']
    x14 = session.xyDataObjects['XYData-15']
    x15 = session.xyDataObjects['XYData-16']
    session.xyReportOptions.setValues(numDigits=9, numberFormat=SCIENTIFIC)
    session.writeXYReport(fileName='abaqus.rpt', xyData=(x0, x1, x2, x3, x4, x5, 
        x6, x7, x8, x9, x10, x11, x12, x13, x14, x15))


def Macro2():
    import visualization
    import xyPlot
    import displayGroupOdbToolset as dgo
    odb = session.odbs['/home/lcharleux/Documents/Programmation/Python/Modules/hardness/doc/tutorials/workdir/indentation_2D.odb']
    session.writeFieldReport(fileName='abaqus.rpt', append=ON, 
        sortItem='Element Label', odb=odb, step=1, frame=1, 
        outputPosition=INTEGRATION_POINT, variable=(('S', INTEGRATION_POINT), 
        ))


