import pandas as pd
import numpy as np
import os
from argiope.abq.pypostproc import read_field_report as rfr
from argiope.mesh import read_h5, Field, write_xdmf

odbPath= "#ODBPATH"
indenter_mesh = read_h5("outputs/{0}_indenter_mesh.h5".format(odbPath))
sample_mesh   = read_h5("outputs/{0}_sample_mesh.h5".format(odbPath))


# FILES LISTING
files = os.listdir("reports/")

for path in files:
#HISTORY OUTPUTS

# FIELD OUTPUTS
  if path.endswith(".frpt"):
    print "#LOADING: " + path
    instance = path.split("instance")[1][1:].split("_step-")[0]
    info = {"tag": path[:-5], "position": "Nodal"}
    data = rfr("reports/" + path)
    field = Field(info, data)
    if instance == "I_INDENTER":
      indenter_mesh.add_field(tag = info["tag"], field = field)
    elif instance == "I_SAMPLE":
      sample_mesh.add_field(tag = info["tag"], field = field) 
   
indenter_mesh.save()
sample_mesh.save()
write_xdmf(sample_mesh,
           "outputs/{0}_sample_mesh.xdmf".format(odbPath), 
           dataformat = "XML")
write_xdmf(indenter_mesh,
           "outputs/{0}_indenter_mesh.xdmf".format(odbPath), 
           dataformat = "XML")           

