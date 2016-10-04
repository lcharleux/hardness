import numpy as np
import pandas as pd
from argiope import mesh as Mesh
import argiope
import os, subprocess, inspect
MODPATH = os.path.dirname(inspect.getfile(argiope))


def sample_mesh_2D(gmsh_path, workdir, lx = 1., ly = 1., r1 = 2., r2 = 1000., Nx = 32, Ny = 16, lc1 = 0.08, lc2 = 200., geoPath = "dummy"):
  """
  Builds an indentation mesh.
  """
  geo = open(MODPATH + "/templates/indentation/indentation_mesh_2D.geo").read()
  geo = geo.replace("#LX", str(lx))
  geo = geo.replace("#LY", str(ly))
  geo = geo.replace("#R1", str(r1))
  geo = geo.replace("#R2", str(r2))
  geo = geo.replace("#NX", str(Nx))
  geo = geo.replace("#NY", str(Ny))
  geo = geo.replace("#LC1", str(lc1))
  geo = geo.replace("#LC2", str(lc2))
  open(workdir + geoPath + ".geo", "w").write(geo)
  p = subprocess.Popen("{0} -2 {1}".format(gmsh_path, geoPath + ".geo"), cwd = workdir, shell=True, stdout = subprocess.PIPE)
  trash = p.communicate()
  mesh = Mesh.read_msh(workdir + geoPath + ".msh")
  mesh.element_set_to_node_set(tag = "SURFACE")
  mesh.element_set_to_node_set(tag = "BOTTOM")
  mesh.element_set_to_node_set(tag = "AXIS")
  del mesh.elements.sets["SURFACE"]
  del mesh.elements.sets["BOTTOM"]
  del mesh.elements.sets["AXIS"]
  mesh.elements.data = mesh.elements.data[mesh.elements.data.etype != "Line2"] 
  mesh.node_set_to_surface("SURFACE")
  mesh.elements.add_set("ALL_ELEMENTS", mesh.elements.data.index)  
  return mesh
  
  
def indentation_input(sample_mesh, path = None):
  """
  Returns a indentation INP file.
  """
  pattern = open(MODPATH + "/templates/indentation/indentation.inp").read()
  element_map = {"Tri3":  "CAX3", 
                 "Quad4": "CAX4", }
  pattern = pattern.replace("#SAMPLE_MESH", 
                            sample_mesh.to_inp(element_map = element_map))
  if path == None:            
    return pattern
  else:
    open(path, "wb").write(pattern)  
    
    
def indentation_abqpostproc(workdir, path, odbPath, histPath, contactPath, fieldPath):
  """
  Writes the abqpostproc file in the workdir.
  """
  pattern = open(MODPATH + "/templates/indentation/abqpostproc.py").read()
  pattern = pattern.replace("#ODBPATH",     odbPath)
  pattern = pattern.replace("#HISTPATH",    histPath)
  pattern = pattern.replace("#CONTACTPATH", contactPath)
  pattern = pattern.replace("#FIELDPATH",   fieldPath)
  open(workdir + path, "wb").write(pattern)
      
def indentation_pypostproc(path, workdir, histPath, contactPath, fieldPath):
  """
  Writes the pypostproc file in the workdir.
  """
  pattern = open(MODPATH + "/templates/indentation/pypostproc.py").read() 
  pattern = pattern.replace("#HISTPATH",    histPath + ".rpt")
  pattern = pattern.replace("#CONTACTPATH", contactPath + ".rpt")
  pattern = pattern.replace("#FIELDPATH",   fieldPath + ".rpt")
  open(workdir + path, "wb").write(pattern)     
      
      
