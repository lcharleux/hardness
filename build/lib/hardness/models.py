import numpy as np
import pandas as pd
#from argiope import mesh as Mesh
import argiope, hardness
import os, subprocess, inspect
from string import Template

# PATH TO MODULE
import hardness
MODPATH = os.path.dirname(inspect.getfile(hardness))


################################################################################
# MODEL DEFINITION
################################################################################
class Indentation2D(argiope.models.Model, argiope.utils.Container):
  """
  2D indentation class.
  """
    
  def write_input(self):
    """
    Writes the input file in the chosen format.
    """
    hardness.models.indentation_2D_input(sample_mesh = self.parts["sample"],
                                   indenter_mesh = self.parts["indenter"],
                                   steps = self.steps,
                                   materials = self.materials,
                                   solver = self.solver,
                                   path = "{0}/{1}.inp".format(self.workdir,
                                                               self.label))
                                   
                                   
    
  def write_postproc(self):
    """
    Writes the prosproc scripts for the chosen solver.
    """
    if self.solver == "abaqus":
      hardness.postproc.indentation_abqpostproc(
          path = "{0}/{1}_abqpp.py".format(
              self.workdir,
              self.label),
          label = self.label,    
          solver= self.solver)
  
  def postproc(self):
     """
     Runs the whole post proc.
     """
     self.write_postproc()
     self.run_postproc()
     #HISTORY OUTPUTS
     hist_path = self.workdir + "/reports/{0}_hist.hrpt".format(self.label)
     if os.path.isfile(hist_path):
       hist = argiope.abq.pypostproc.read_history_report(
            hist_path, steps = self.steps, x_name = "t") 
       hist["F"] = hist.CF + hist.RF
       self.data["history"] = hist
     # FIELD OUTPUTS
     files = os.listdir(self.workdir + "reports/")
     files = [f for f in files if f.endswith(".frpt")]
     files.sort()
     for path in files:
       field = argiope.abq.pypostproc.read_field_report(
                           self.workdir + "reports/" + path)
       if field.part == "I_SAMPLE":
         self.parts["sample"].mesh.fields.append(field)
       if field.part == "I_INDENTER":
         self.parts["indenter"].mesh.fields.append(field)
################################################################################
# MESH PROCESSING
################################################################################
def process_2D_sample_mesh(part):
  """
  Processes a 2D mesh, indenter or sample
  """
  mesh = part.mesh
  element_map = part.element_map
  material_map = part.material_map
  mesh.elements[("sets", "ALL_ELEMENTS", "")] = True
  mesh.nodes[("sets", "ALL_NODES")] = True
  mesh.element_set_to_node_set(tag = "SURFACE")
  mesh.element_set_to_node_set(tag = "BOTTOM")
  mesh.element_set_to_node_set(tag = "AXIS")
  del mesh.elements[("sets", "SURFACE", "")]
  del mesh.elements[("sets", "BOTTOM", "")]
  del mesh.elements[("sets", "AXIS", "")]
  mesh.elements = mesh.elements.loc[mesh.space() == 2] 
  mesh.node_set_to_surface("SURFACE")
  if element_map != None:
    mesh = element_map(mesh)
  if material_map != None:
    mesh = material_map(mesh)
  return mesh                                      


def process_2D_indenter_mesh(part):
  """
  Processes a raw gmsh 2D indenter mesh 
  """
  mesh = part.mesh
  element_map = part.element_map
  material_map = part.material_map
  mesh = process_2D_sample_mesh(part)
  x, y = mesh.nodes.coords.x.values, mesh.nodes.coords.y.values
  mesh.nodes[("sets","TIP_NODE")] = (x == 0) * (y == 0)
  mesh.nodes[("sets","REF_NODE")] = (x == 0) * (y == y.max())
  if part.rigid == False:
    mesh.nodes.loc[:, ("sets", "RIGID_NODES")] = (
         mesh.nodes.sets["BOTTOM"])
  else: 
    mesh.nodes.loc[:, ("sets", "RIGID_NODES")] = (
         mesh.nodes.sets["ALL_NODES"])
  return mesh        


################################################################################
# PARTS
################################################################################  


'''
def conical_indenter_mesh_2D(gmsh_path, workdir, psi= 70.29, 
                             r1 = 1., r2 = 10., r3 = 100., 
                             lc1 = 0.1, lc2 = 20., 
                             rigid = False, 
                             geoPath = "conical_indenter_2D", 
                             algorithm = "delquad", 
                             **kwargs):
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
  return process_2D_indenter(mesh, **kwargs)
'''
class Sample(argiope.models.Part):
  pass
   
class Sample2D(Sample):
  """
  A 2D indentation mesh.
  """
  def __init__(self, lx = 1., ly = 1., 
                     r1 = 2., r2 = 1000., 
                     Nx = 32, Ny = 32,
                     Nr = 16, Nt = 8, 
                     **kwargs):
    self.lx = lx
    self.ly = ly
    self.r1 = r1
    self.r2 = r2
    self.Nx = Nx
    self.Ny = Ny
    self.Nr = Nr
    self.Nt = Nt
    super().__init__(**kwargs)
    
  def preprocess_mesh(self):
    lx = self.lx
    ly = self.ly
    r1 = self.r1
    r2 = self.r2
    Nx = self.Nx
    Ny = self.Ny
    Nr = self.Nr
    Nt = self.Nt
    
    q1 = (r2/r1)**(1./Nr) 
    lcx, lcy = lx / Nx, ly / Ny
    geo = Template(
          open(MODPATH + 
          "/templates/models/indentation_2D/indentation_mesh_2D.geo").read())
    geo = geo.substitute(
          lx = lx,
          ly = ly,
          r1 = r1,
          r2 = r2,
          Nx = Nx,
          Ny = Ny,
          Nr = Nr,
          Nt = Nt,
          q1 = q1)
    open(self.workdir + self.file_name + ".geo", "w").write(geo)

  def postprocess_mesh(self):
    self.mesh = process_2D_sample_mesh(self)

class SampleFibre2D(Sample):
  """
  A 2D sample with a vertical fibre indentation mesh.
  """
  def __init__(self, Rf = 1., 
                     ly1 = 1., ly2 = 10.,
                     Nx = 16, Ny = 8,
                     Nr = 16, Nt = None,
                     **kwargs):
    if Nt == None: Nt = np.pi / 2. * Ny
    self.Rf = Rf
    self.ly1 = ly1
    self.ly2 = ly2
    self.Nx = Nx
    self.Ny = Ny
    self.Nr = Nr
    self.Nt = Nt
    super().__init__(**kwargs)
    
  def preprocess_mesh(self):
    Rf = self.Rf
    ly1 = self.ly1
    ly2 = self.ly2
    Nx = self.Nx
    Ny = self.Ny
    Nr = self.Nr
    Nt = self.Nt
    geo = Template(
       open(MODPATH + 
       "/templates/models/indentation_2D/indentation_fibre_mesh_2D.geo").read())
    geo = geo.substitute(
          Rf = Rf,
          ly1 = ly1,
          ly2 = ly2,
          Nx = Nx,
          Ny = Ny,
          Nr = Nr,
          Nt = Nt)
    open(self.workdir + self.file_name + ".geo", "w").write(geo)

  def postprocess_mesh(self):
    self.mesh = process_2D_sample_mesh(self)

class Indenter(argiope.models.Part):
  """
  A generic indenter metaclass.
  """
  def __init__(self, rigid = True, **kwargs):
    self.rigid = rigid
    super().__init__(**kwargs)

class Indenter2D(Indenter):
  """
  A generic 2D indenter metaclass.
  """
  def postprocess_mesh(self):
    self.mesh = process_2D_indenter_mesh(self)
  
class SpheroconicalIndenter2D(Indenter2D):
  """
  A spheroconical indenter class.
  """
  
  def __init__(self, psi= 70.29, R = 1., 
                     r1 = 1., r2 = 10., r3 = 100., 
                     lc1 = 0.1, lc2 = 20.,
                     **kwargs):
    self.psi = psi
    self.R = R
    self.r1 = r1
    self.r2 = r2
    self.r3 = r3
    self.lc1 = lc1
    self.lc2 = lc2
    super().__init__(**kwargs)
  
  def preprocess_mesh(self):                 
    psi = self.psi
    R = self.R
    r1 = self.r1
    r2 = self.r2
    r3 = self.r3
    lc1 = self.lc1
    lc2 = self.lc2
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
    open(self.workdir + self.file_name + ".geo", "w").write(geo)
  
  
  
  

  
################################################################################
# 2D STEP
################################################################################  
class Step2D:
  """
  A general purpose 2D indentation step.
  """
  def __init__(self, control_type = "disp", 
                     name = "STEP", 
                     duration = 1., 
                     nframes = 100,
                     kind = "fixed", 
                     controlled_value = .1,
                     min_frame_duration = 1.e-8,
                     field_output_frequency = 99999,
                     solver = "abaqus"):
    self.control_type = control_type
    self.name = name  
    self.duration = duration
    self.nframes = nframes
    self.kind = kind  
    self.controlled_value = controlled_value
    self.min_frame_duration = min_frame_duration
    self.field_output_frequency = field_output_frequency
    self.solver = solver
                     
  def get_input(self):
    control_type = self.control_type 
    name = self.name 
    duration = self.duration
    nframes = self.nframes
    kind = self.kind 
    controlled_value = self.controlled_value
    min_frame_duration = self.min_frame_duration
    solver = self.solver
    rootPath = "/templates/models/indentation_2D/steps/"
    if solver == "abaqus":
      if kind == "fixed":
        if control_type == "disp":
          pattern = rootPath + "indentation_2D_step_disp_control_fixed.inp"
        if control_type == "force":
          pattern = rootPath + "indentation_2D_step_load_control_fixed.inp"  
        pattern = Template(open(MODPATH + pattern).read())
                
        return pattern.substitute(NAME = name,
                           CONTROLLED_VALUE = controlled_value,
                           DURATION = duration,
                           FRAMEDURATION = float(duration) / nframes, 
                           FIELD_OUTPUT_FREQUENCY = self.field_output_frequency)
      if kind == "adaptative":
        if control_type == "disp":
          pattern = rootPath + "indentation_2D_step_disp_control_adaptative.inp"
        if control_type == "force":
          pattern = rootPath + "indentation_2D_step_load_control_adaptative.inp"  
        pattern = Template(open(MODPATH + pattern).read())
                
        return pattern.substitute(NAME = name,
                           CONTROLLED_VALUE = controlled_value,
                           DURATION = duration,
                           FRAMEDURATION = float(duration) / nframes, 
                           MINFRAMEDURATION = min_frame_duration,
                           FIELD_OUTPUT_FREQUENCY = self.field_output_frequency)                           

################################################################################
# 2D ABAQUS INPUT FILE
################################################################################  
def indentation_2D_input(sample_mesh, 
                         indenter_mesh,
                         steps, 
                         materials,
                         path = None, 
                         element_map = None, 
                         solver = "abaqus",
                         make_mesh = True):
  """
  Returns a indentation input file.
  """
  if make_mesh:
    sample_mesh.make_mesh()
    indenter_mesh.make_mesh()
    
  if solver == "abaqus":
    pattern = Template(
        open(MODPATH + "/templates/models/indentation_2D/indentation_2D.inp")
        .read())
    
    pattern = pattern.substitute(
        SAMPLE_MESH = sample_mesh.mesh.write_inp(),
        INDENTER_MESH = indenter_mesh.mesh.write_inp(),
        STEPS = "".join([step.get_input() for step in steps]),
        MATERIALS = "\n".join([m.write_inp() for m in materials]) )
  if path == None:            
    return pattern
  else:
    open(path, "w").write(pattern)  
