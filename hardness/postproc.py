import numpy as np
import pandas as pd
from argiope import mesh as Mesh
import argiope
import os, subprocess, inspect

# PATH TO MODULE
import hardness
MODPATH = os.path.dirname(inspect.getfile(hardness))

def indentation_abqpostproc(workdir, odbPath):
  """
  Writes the abqpostproc file in the workdir.
  """
  pattern = open(MODPATH + "/templates/postproc/abqpostproc.py").read()
  pattern = pattern.replace("#ODBPATH",     odbPath)
  open(workdir + odbPath + "_abqpostproc.py", "wb").write(pattern)
      
def indentation_pypostproc(workdir, odbPath):
  """
  Writes the pypostproc file in the workdir.
  """
  pattern = open(MODPATH + "/templates/postproc/pypostproc.py").read() 
  pattern = pattern.replace("#ODBPATH",    odbPath)
  open(workdir + odbPath + "_pypostproc.py", "wb").write(pattern)     
