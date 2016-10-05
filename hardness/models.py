import numpy as np
import pandas as pd
from argiope import mesh as Mesh
import argiope
import os, subprocess, inspect

# PATH TO MODULE
import hardness
MODPATH = os.path.dirname(inspect.getfile(hardness))


def sample_mesh_2D(gmsh_path, workdir, lx = 1., ly = 1., r1 = 2., r2 = 1000., Nx = 32, Ny = 16, lc1 = 0.08, lc2 = 200., geoPath = "sample_mesh_2D"):
  """
  Builds an indentation mesh.
  """
  geo = open(MODPATH + "/templates/models/indentation_mesh_2D.geo").read()
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
  
def conical_indenter_mesh_2D(gmsh_path, workdir, psi= 70.29, r1 = 1., r2 = 10., r3 = 100., lc1 = 0.1, lc2 = 20., rigid = False, geoPath = "conical_indenter_2D"):
  """
  Builds a conical indenter mesh.
  """
  # Some high lvl maths...
  psi = np.radians(psi)
  x2 = r3 * np.sin(psi)
  y2 = r3 * np.cos(psi)
  y3 = r3 
  # Template filling  
  geo = open(MODPATH + "/templates/models/conical_indenter_mesh_2D.geo").read()
  geo = geo.replace("#LC1", str(lc1))
  geo = geo.replace("#LC2", str(lc2))
  geo = geo.replace("#R1",  str(r1))
  geo = geo.replace("#R2",  str(r2))
  geo = geo.replace("#X2",  str(x2))
  geo = geo.replace("#Y2",  str(y2))
  geo = geo.replace("#Y3",  str(y3))
  
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
  if rigid == False:
    mesh.nodes.add_set("RIGID_NODES", mesh.nodes.sets["BOTTOM"]) 
  else:
    mesh.nodes.add_set("RIGID_NODES", mesh.nodes.sets["ALL_ELEMENTS"])
  mesh.nodes.add_set_by_func("TIP_NODE", lambda x, y, z, labels: ((x == 0.) * (y == 0.)) == True )
  mesh.nodes.add_set_by_func("REF_NODE", lambda x, y, z, labels: ((x == 0.) * (y == y.max())) == True )      
  return mesh  
  
  
def indentation_input(sample_mesh, indenter_mesh, path = None, element_map = None):
  """
  Returns a indentation INP file.
  """
  pattern = open(MODPATH + "/templates/models/indentation.inp").read()
  if element_map == None:
    element_map = {"Tri3":  "CAX3", 
                   "Quad4": "CAX4", }
  pattern = pattern.replace("#SAMPLE_MESH", 
                            sample_mesh.to_inp(element_map = element_map))
  pattern = pattern.replace("#INDENTER_MESH", 
                            indenter_mesh.to_inp(element_map = element_map))                          
  if path == None:            
    return pattern
  else:
    open(path, "wb").write(pattern)  
