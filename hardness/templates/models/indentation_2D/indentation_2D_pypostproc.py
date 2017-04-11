import pandas as pd
import numpy as np
import os
from argiope.abq.pypostproc import read_field_report as rfr
from argiope.mesh import read_h5, Field, write_xdmf

simName= "${simName}"
#indenter_mesh = read_h5("outputs/{0}_indenter_mesh.h5".format(simName))
#sample_mesh   = read_h5("outputs/{0}_sample_mesh.h5".format(simName))


# FILES LISTING
files = os.listdir("reports/")

for path in files:
#HISTORY OUTPUTS

# FIELD OUTPUTS
  if path.endswith(".frpt"):
    print "#LOADING: " + path
    sname, d    = path.split("_instance-")
    instance, d = d.split("_step-")
    step, d    = d.split("_frame-")
    frame, d    = d.split("_var-")
    var, d    = d.split(".")
    info = {"tag": "step-{0}_frame-{1}_var-{2}".format(step, frame, var), "position": "Nodal"}
    data = rfr("reports/" + path)
    field = Field(info, data)
    if instance == "I_INDENTER":
      indenter_mesh.add_field(tag = info["tag"], field = field)
    elif instance == "I_SAMPLE":
      sample_mesh.add_field(tag = info["tag"], field = field) 
   
indenter_mesh.save()
sample_mesh.save()
write_xdmf(sample_mesh,
           "outputs/{0}_sample_mesh".format(simName), 
           dataformat = "XML")
write_xdmf(indenter_mesh,
           "outputs/{0}_indenter_mesh".format(simName), 
           dataformat = "XML")           

