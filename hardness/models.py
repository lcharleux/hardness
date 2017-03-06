import numpy as np
import pandas as pd
from argiope import mesh as Mesh
import argiope
import os, subprocess, inspect
from string import Template

# PATH TO MODULE
import hardness
MODPATH = os.path.dirname(inspect.getfile(hardness))


def sample_mesh_2D(gmsh_path, workdir, 
                   lx = 1., ly = 1., 
                   r1 = 2., r2 = 1000., 
                   Nx = 32, Ny = 16, 
                   lc1 = 0.08, lc2 = 200., 
                   geoPath = "sample_mesh_2D", 
                   algorithm = "del2d"):
  """
  Builds an indentation mesh.
  """
  geo = Template(
        open(MODPATH + "/templates/models/indentation_2D/indentation_mesh_2D.geo").read())
  geo = geo.substitute(
        lx = lx,
        ly = ly,
        r1 = r1,
        r2 = r2,
        Nx = Nx,
        Ny = Ny,
        lc1 = lc1,
        lc2 = lc2)
  open(workdir + geoPath + ".geo", "w").write(geo)
  p = subprocess.Popen("{0} -2 -algo {1} {2}".format(gmsh_path, 
                                                     algorithm,
                                                     geoPath + ".geo"), 
                       cwd = workdir, shell=True, stdout = subprocess.PIPE)  
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
  
def conical_indenter_mesh_2D(gmsh_path, workdir, psi= 70.29, 
                             r1 = 1., r2 = 10., r3 = 100., 
                             lc1 = 0.1, lc2 = 20., 
                             rigid = False, 
                             geoPath = "conical_indenter_2D", 
                                   algorithm = "delquad"):
  """
  Builds a conical indenter mesh.
  """
  # Some high lvl maths...
  psi = np.radians(psi)
  x2 = r3 * np.sin(psi)
  y2 = r3 * np.cos(psi)
  y3 = r3 
  # Template filling  
  geo = Template(
        open(MODPATH + "/templates/models/indentation_2D/conical_indenter_mesh_2D.geo").read())
  geo = geo.substitute(
    lc1 = lc1,
    lc2 = lc2,
    r1  = r1,
    r2  = r2,
    x2  = x2,
    y2  = y2,
    y3  = y3)
  open(workdir + geoPath + ".geo", "w").write(geo)
  p = subprocess.Popen("{0} -2 -algo {1} {2}".format(gmsh_path, 
                                                     algorithm,
                                                     geoPath + ".geo"), 
                       cwd = workdir, shell=True, stdout = subprocess.PIPE)
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
  
def spheroconical_indenter_mesh_2D(gmsh_path, workdir, psi= 70.29, R = 1., 
                                   r1 = 1., r2 = 10., r3 = 100., 
                                   lc1 = 0.1, lc2 = 20., 
                                   rigid = False, 
                                   geoPath = "spheroconical_indenter_2D", 
                                   algorithm = "delquad"):
  """
  Builds a spheroconical indenter mesh.
  """
  # Some high lvl maths...
  psi = np.radians(psi)
  x2 = R  * np.cos(psi)
  y2 = R  * (1. - np.sin(psi))
  x3 = r3 * np.sin(psi)
  dh = R * (1. / np.sin(psi) - 1.)
  y3 = r3 * np.cos(psi) - dh
  y4 = r3 - dh
  y5 = R 
  y6 = -dh
  # Template filling  
  geo = Template(
        open(MODPATH + "/templates/models/indentation_2D/spheroconical_indenter_mesh_2D.geo").read())
  geo = geo.substitute(
     lc1 = lc1,
     lc2 = lc2,
     r1 = r1,
     r2 = r2,
     x2 = x2,
     y2 = y2,
     x3 = x3,
     y3 = y3,
     y4 = y4,
     y5 = y5,
     y6 = y6)
  open(workdir + geoPath + ".geo", "w").write(geo)
  p = subprocess.Popen("{0} -2 -algo {1} {2}".format(gmsh_path, 
                                                     algorithm,
                                                     geoPath + ".geo"), 
                       cwd = workdir, shell=True, stdout = subprocess.PIPE)
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
  
  
def indentation_2D_step_input(control_type = "disp", 
                              name = "STEP", 
                              duration = 1., 
                              nframes = 100, 
                              controlled_value = .1,
                              solver = "abaqus"):
  if solver == "abaqus":
    if control_type == "disp":
      pattern = "/templates/models/indentation_2D/indentation_2D_step_disp_control.inp"
    if control_type == "force":
      pattern = "/templates/models/indentation_2D/indentation_2D_step_load_control.inp"  
    pattern = Template(open(MODPATH + pattern).read())
            
    return pattern.substitute(NAME = name,
                             CONTROLLED_VALUE = controlled_value,
                             DURATION = duration,
                             FRAMEDURATION = float(duration) / nframes, )
  
def indentation_2D_input(sample_mesh, 
                         indenter_mesh,
                         steps, 
                         path = None, 
                         element_map = None, 
                         solver = "abaqus"):
  """
  Returns a indentation input file.
  """
  if solver == "abaqus":
    pattern = Template(
            open(MODPATH + "/templates/models/indentation_2D/indentation_2D.inp").read())
    
    if element_map == None:
      element_map = {"Tri3":  "CAX3", 
                     "Quad4": "CAX4", }
    pattern = pattern.substitute(
        SAMPLE_MESH = sample_mesh.to_inp(element_map = element_map),
        INDENTER_MESH = indenter_mesh.to_inp(element_map = element_map),
        STEPS = "".join(steps))
  if path == None:            
    return pattern
  else:
    open(path, "w").write(pattern)  
