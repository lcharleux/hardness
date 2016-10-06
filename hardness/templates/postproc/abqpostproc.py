# -*- coding: mbcs -*-
from abaqus import *
from abaqusConstants import *
import visualization, xyPlot
import displayGroupOdbToolset as dgo
import __main__

# SETTINGS
odbPath     = "#ODBPATH"

# REPORT FOLDER SETUP
try:
  os.mkdir("reports")
except:
  pass  

# DATABASE SETUP
o1 = session.openOdb(name = odbPath + ".odb")
session.viewports['Viewport: 1'].setValues(displayedObject=o1)
session.xyReportOptions.setValues(numDigits=9, numberFormat=SCIENTIFIC)
odb = session.odbs[odbPath + ".odb"]

# SIMULATION STATUS 
job_completed = (odb.diagnosticData.jobStatus == JOB_STATUS_COMPLETED_SUCCESSFULLY)
open(odbPath + "_completed.txt", "wb").write(str(job_completed))

if job_completed:
  stepKeys = odb.steps.keys()
  
  # CONTACT DATA
  surface_nodes = [n.label for n in  odb.rootAssembly.instances["I_SAMPLE"].nodeSets["SURFACE"].nodes]
  tags =  ["Coordinates: COOR1 PI: I_SAMPLE Node {0} in NSET SURFACE".format(l) for l in surface_nodes]
  tags += ["Coordinates: COOR2 PI: I_SAMPLE Node {0} in NSET SURFACE".format(l) for l in surface_nodes]
  tags += ["Contact pressure: CPRESS   ASSEMBLY_I_SAMPLE_SURFACE/ASSEMBLY_I_INDENTER_SURFACE PI: I_SAMPLE Node {0}".format(l) for l in surface_nodes]
  cols =  ["COOR1_n{0}".format(l) for l in surface_nodes]
  cols += ["COOR2_n{0}".format(l) for l in surface_nodes]
  cols += ["CPRESS_n{0}".format(l) for l in surface_nodes]
  contDict = {}
  for i in xrange(len(tags)): contDict[cols[i]] = tags[i] 
  contData = [session.XYDataFromHistory(
                  name= key, 
                  odb=odb, 
                  outputVariableName= value, 
                  steps = stepKeys)
          for key, value in contDict.iteritems()] 
 
  session.writeXYReport(fileName="reports/" + odbPath + "_contact.rpt", 
                        xyData = contData)
  
   
  # HISTORY OUTPUTS
  ref_node = [n.label for n in  odb.rootAssembly.instances["I_INDENTER"].nodeSets["REF_NODE"].nodes][0]
  tip_node = [n.label for n in  odb.rootAssembly.instances["I_INDENTER"].nodeSets["TIP_NODE"].nodes][0]
  histDict = {
              "Wtot":"External work: ALLWK for Whole Model",
              "Wf"  :"Frictional dissipation: ALLFD for Whole Model",
              "Wps" :"Plastic dissipation: ALLPD PI: I_SAMPLE in ELSET ALL_ELEMENTS",
              "Wei" :"Strain energy: ALLSE PI: I_INDENTER in ELSET ALL_ELEMENTS",
              "Wes" :"Strain energy: ALLSE PI: I_SAMPLE in ELSET ALL_ELEMENTS",
              "F"   :"Reaction force: RF2 PI: I_INDENTER Node {0} in NSET REF_NODE".format(ref_node),
              "dtip":"Spatial displacement: U2 PI: I_INDENTER Node {0} in NSET TIP_NODE".format(tip_node),
              "dtot":"Spatial displacement: U2 PI: I_INDENTER Node {0} in NSET REF_NODE".format(ref_node),   
             }
  
  histData = [session.XYDataFromHistory(
                  name= key, 
                  odb=odb, 
                  outputVariableName= value, 
                  steps = stepKeys)
          for key, value in histDict.iteritems()] 
 
  session.writeXYReport(fileName="reports/" + odbPath + "_hist.rpt", 
                        xyData = histData)


  # FIELD OUTPUTS
  nf = NumberFormat(numDigits=9, precision=0, format=SCIENTIFIC)
  session.fieldReportOptions.setValues(
          printTotal=OFF, 
          printMinMax=OFF, 
          numberFormat=nf)
  instances = ("I_SAMPLE", "I_INDENTER")
  fields = {"S":  
                  (('S', INTEGRATION_POINT, 
                      ((COMPONENT, 'S11'),  
                      (COMPONENT, 'S22'), 
                      (COMPONENT, 'S33'), 
                      (COMPONENT, 'S12'), 
                    )),)
           }
  
  for instance in instances:
    leaf = dgo.LeafFromPartInstance(partInstanceName = instance)
    session.viewports['Viewport: 1'].odbDisplay.displayGroup.replace(leaf=leaf)
    for stepNum in xrange(len(stepKeys)):
      stepKey = stepKeys[stepNum]
      frames  = odb.steps[stepKey].frames
      nFrames = len(frames)
      for frameNum in xrange(nFrames):
        frame = frames[frameNum]
        for fieldKey, field in fields.iteritems():
          session.writeFieldReport(
                fileName       = "reports/{0}_instance={1}_step={2}_frame={3}_var={4}.rpt".format(
                    odbPath,
                    instance,     
                    stepKey,
                    frameNum,
                    fieldKey,), 
                append         = OFF, 
                sortItem       = 'Node Label',
                odb            = odb, 
                step           = stepNum, 
                frame          = frameNum, 
                outputPosition = NODAL, 
                variable       = field)


