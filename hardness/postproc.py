import numpy as np
import pandas as pd
from argiope import mesh as Mesh
import argiope
import os, subprocess, inspect
from string import Template

# PATH TO MODULE
import hardness
MODPATH = os.path.dirname(inspect.getfile(hardness))

def indentation_abqpostproc(path, label, solver = "abaqus"):
  """
  Writes the abqpostproc file in the workdir.
  """
  if solver == "abaqus":
    pattern = Template(
          open(MODPATH + "/templates/models/indentation_2D/indentation_2D_abqpostproc.py").read())
    pattern = pattern.substitute(simName = label)
    open(path, "w").write(pattern)
      
def indentation_pypostproc(path):
  """
  Writes the pypostproc file in the workdir.
  """
  pattern = Template(
        open(MODPATH + "/templates/models/indentation_2D/indentation_2D_pypostproc.py").read()) 
  pattern = pattern.substitute(simName = simName)
  open(path, "w").write(pattern)     
