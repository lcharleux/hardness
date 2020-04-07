import numpy as np
import pandas as pd
#from argiope import mesh as Mesh
import argiope, hardness
import os, subprocess, inspect
from string import Template
import gmsh
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
  
  def __init__(self, friction = 0., *args, **kwargs):
    self.friction = friction
    argiope.models.Model.__init__(self, *args, **kwargs)
    
  def write_input(self):
    """
    Writes the input file in the chosen format.
    """
    hardness.models.indentation_2D_input(sample_mesh = self.parts["sample"],
                                   indenter_mesh = self.parts["indenter"],
                                   steps = self.steps,
                                   materials = self.materials,
                                   friction = self.friction,
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
     completed_path = self.workdir + "/{0}_completed.txt".format(self.label)
     if os.path.isfile(completed_path):
        completed_value = open(completed_path).read().strip().lower()
        if completed_value == "true":
            print("# SIMULATION COMPLETED: EXTRACTING OUTPUTS")
            hist_path = self.workdir + "/reports/{0}_hist.hrpt".format(self.label)
            raw_hist = argiope.abq.pypostproc.read_history_report(
                hist_path, steps = self.steps, x_name = "t") 
            #hist = raw_hist[["t", "Wes", "Wei", "Wps", "Wpi", "Wf", "Wtot",]].copy()
            hist = pd.DataFrame()
            hist[("time", "")] = raw_hist.t
            hist[("step", "")] = raw_hist.step
            # FORCES
            hist[("force", "F")] = -(raw_hist.CF + raw_hist.RF)
            # DISPLACEMENTS
            hist[("disp", "htot")] = -raw_hist.dtot
            hist[("disp", "hsamp")] = -raw_hist.dtip
            hist[("disp", "hind")] = hist[("disp", "htot")] - hist[("disp", "hsamp")]
            # ENERGIES
            hist[("energies", "Etot")] = raw_hist.Etot
            hist[("energies", "Eint")] = raw_hist.Eint
            hist[("energies", "Wext")] = raw_hist.Wext
            hist[("energies", "Wart")] = raw_hist.Wart
            hist[("energies", "Wfric")] = raw_hist.Wfric
            hist[("energies", "Wels")] = raw_hist.Welast_s
            hist[("energies", "Wels")] = raw_hist.Welast_i
            #hist["ht"] = -raw_hist.dtot
            #hist["hs"] = -raw_hist.dtip
            #hist["hi"] = hist.ht - hist.hs 
            hist[("contact", "Aw_carea")] = raw_hist.Carea
            
            # CONTACT HISTORY
            contact_path = self.workdir + "/reports/{0}_contact.hrpt".format(
                           self.label)
            contact_data = argiope.abq.pypostproc.read_history_report(contact_path, 
                                                                 steps=self.steps)
            cols = contact_data.columns.values
            coor1_cols = sorted([c for c in cols if c.startswith("COOR1")])
            coor2_cols = sorted([c for c in cols if c.startswith("COOR2")])
            cpress_cols = sorted([c for c in cols if c.startswith("CPRESS")])
            coor1 = contact_data[coor1_cols].values
            order = np.argsort(coor1[0])
            coor2 = contact_data[coor2_cols].values
            cpress = contact_data[cpress_cols].values
            coor1 = coor1[:, order]
            coor2 = coor2[:, order]
            cpress = cpress[:, order]
            ind = np.arange(coor1.shape[0])
            mask = np.outer(np.ones(coor1.shape[0]) , 
                            np.arange(coor1.shape[1])).astype(np.int32)
            mask *= cpress > 0.
            mask = mask.max(axis =1)
            upper_mask = np.clip(mask+1, 0, coor1.shape[1])
            rc_lower = coor1[ind, mask]
            rc_upper = coor1[ind, upper_mask]
            rc_mid =  (rc_upper + rc_lower)/2.
            zc_lower = coor2[ind, mask]
            zc_upper = coor2[ind, upper_mask]
            zc_mid =  (zc_upper + zc_lower)/2.

            hist[("contact", "rc_lower")] = rc_lower
            hist[("contact", "rc_upper")] = rc_upper
            hist[("contact", "rc_mid")] = rc_mid
            hist[("contact", "zc_lower")] = zc_lower 
            hist[("contact", "zc_upper")] = zc_upper 
            hist[("contact", "zc_mid")] = zc_mid 
            hist.columns = pd.MultiIndex.from_tuples(hist.columns)  
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


class ConicalIndentation2D(Indentation2D):
    """
    A class dedicated to 2D conical indentation.
    """
    def outputs(self):
        return 
        #self.         
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


class ConicalIndenter2D(Indenter2D):
  """
  A spheroconical indenter class.
  """
  
  def __init__(self, psi= 70.29, 
                     r1 = 1., r2 = 10., r3 = 100., 
                     lc1 = 0.1, lc2 = 20.,
                     **kwargs):
    self.psi = psi
    self.r1 = r1
    self.r2 = r2
    self.r3 = r3
    self.lc1 = lc1
    self.lc2 = lc2
    super().__init__(**kwargs)
  
  def preprocess_mesh(self):                 
    psi = self.psi
    r1 = self.r1
    r2 = self.r2
    r3 = self.r3
    lc1 = self.lc1
    lc2 = self.lc2
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
       r1 = r1,
       r2 = r2,
       x2 = x2,
       y2 = y2,
       y3 = y3)
    open(self.workdir + self.file_name + ".geo", "w").write(geo)

  
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
  
# 3D INDENTERS

class Indenter3D(Indenter):
    """
    A generic (but empty) 3D indenter metaclass.
    """
    pass


class PyramidalIndenter3DFull(Indenter3D):
    """
    A Full 3D pyramidal indenter class.

    Args:
    * psi: axis to face angle (in degrees).
    * Nf: number of faces/edges
    * Rs: radius of the outer sphere
    * Rt: tip and edge radius
    * lc1: characteristic element size at tip.
    * lc2: characteristic element size far from the tip.
    * r1: transition radius for the characteristic length. Below r1 from tip, the elements have a constant size. Above r1, the characteristic length is linerarly increased up to lc2 at Rs distance.
    """
    def __init__(self, psi= 65., Nf = 3,
                     Rs = 10., Rt = 1.,
                     blunt = True, 
                     lc1 = 0.05, lc2 = 5., 
                     r1 = 0.5, *args, **kwargs):
        self.psi = psi
        self.Nf = Nf
        self.Rs = Rs
        self.Rt = Rt
        self.r1 = r1
        self.blunt = blunt
        self.lc1 = lc1
        self.lc2 = lc2
        self.r1 = r1
        super().__init__(**kwargs)
    
    def axis_to_edge_angle(self):
        """
        Returns the axis to edge angle in degrees.
        """
        phi = np.degrees(np.arctan(np.tan(np.radians(self.psi)) 
              / np.cos(np.pi / self.Nf))) # Axis vs edge
        return phi
    
    def delta_h(self):
        """
        Returns the missing tip length delta h.
        """
        return self.Rt * (np.sin(np.radians(self.psi))**-1 - 1. )
    
    def make_mesh(self, use_gui = False):
        """
        Generates the mesh.
        """
        # MODEL SETUP
        model = gmsh.model
        factory = model.occ
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6);
        model.add("indenter")

        # DATA
        psi =  np.radians(self.psi) # Axis to face angle
        Nf = self.Nf # Number of faces
        Rs = self.Rs
        Rt = self.Rt
        blunt = self.blunt

        # MATH
        phi = np.radians(self.axis_to_edge_angle())
        deltah = self.delta_h()

        # PYRAMID STUFF
        theta = np.pi/Nf
        factory.addCylinder(0., 0., 0., 0., 0., Rs*2, Rs, 1)
        for i in range(Nf):
            box = factory.addBox(0., -2*Rs, -2*Rs, 4*Rs, 4*Rs, 4*Rs)
            factory.rotate([(3, box)], 0., 0., 0., 0., 1., 0., psi)
            if theta != 0.:
                factory.rotate([(3, box)], 0., 0., 0., 0., 0., 1., theta)
            theta+= 2. * np.pi / Nf
            factory.cut( [(3, 1),], [(3, box),], removeTool = True) 
        fused = [(3,1)]

        # BlUNTING
        if blunt:
            blunt_tags = [(3,5)]
            factory.addSphere(0., 0., 0., Rt, 5) 
            theta = 0.
            for i in range(Nf):
                cyl = factory.addCylinder(0., 0., 0., 0., 0., 2*Rs, Rt, 10)
                factory.rotate([(3, cyl)], 0., 0., 0., 0., 1., 0., phi)
                if theta != 0.:
                    factory.rotate([(3, cyl)], 0., 0., 0., 0., 0., 1., theta)
                theta+= 2. * np.pi / Nf
                blunt_tags = factory.fuse(blunt_tags, [(3, cyl),], 
                                          removeTool = True)[0] 
            factory.translate(blunt_tags, 0., 0., Rt)
            factory.translate([(3, 1)], 0., 0., -deltah)
            cut_tags = factory.cut([(3, 1),], [blunt_tags], 
                                   removeTool = False)[0]
            # FIND WHAT TO REMOVE AND WHAT TO KEEP !
            factory.synchronize()
            volumes = np.array(model.getEntities(3))[:, 1]  
            surfaces = np.array(model.getEntities(2))[:, 1] 
            curves = np.array(model.getEntities(1))[:, 1] 
            points =  np.array(model.getEntities(0))[:, 1] 
            coordinates = np.array([model.getValue( 0, p, []) for p in points])
            tip_point = points[ coordinates[:,2] == coordinates[:,2].min()][0] 
            for tag in cut_tags: 
                points_in_tag =  np.array(model.getBoundary(tag, 
                                 recursive = True))[:, 1]  
                if tip_point in points_in_tag: 
                    factory.remove([tag], recursive = True)
                else:
                    keep_tag = tag
            fused = factory.fuse([keep_tag], [blunt_tags], 
                                 removeTool = True)[0]
           
        # CUT THE BOTTOM
        factory.addSphere(0., 0., 0., Rs, 100) 
        factory.intersect( fused, [(3, 100),])

        # MODEL ANALYSIS
        factory.synchronize()
        volumes = np.array(model.getEntities(3))[:, 1]  
        surfaces = np.array(model.getEntities(2))[:, 1] 
        curves = np.array(model.getEntities(1))[:, 1] 
        points =  np.array(model.getEntities(0))[:, 1] 
        coordinates = np.array([model.getValue( 0, p, []) for p in points])
        top_point = points[ coordinates[:,2] == coordinates[:,2].max()][0] 
        tip_point = points[ coordinates[:,2] == coordinates[:,2].min()][0] 
        points_in_surfaces = [
                np.array(model.getBoundary([(2, s)], recursive = True))[:,1]
                for s in surfaces]
        top_surface = surfaces[np.array([top_point in pis 
                                         for pis in points_in_surfaces])][0]

        # POINTS
        model.addPhysicalGroup(0, [top_point], 1)
        model.setPhysicalName(0, 1, "INDENTER_TOP_POINT")
        model.addPhysicalGroup(0, [tip_point], 2)
        model.setPhysicalName(0, 2, "INDENTER_TIP_POINT")
        # SURFACES
        model.addPhysicalGroup(2, [top_surface], 1)
        model.setPhysicalName(2, 1, "INDENTER_TOP_SURFACE")
        model.addPhysicalGroup(2, [s for s in surfaces if s != top_surface], 2)
        model.setPhysicalName(2, 2, "INDENTER_BOTTOM_SURFACE")
        # VOLUMES
        model.addPhysicalGroup(3, volumes, 1)
        model.setPhysicalName(3, 1, "INDENTER_BULK")
         
        # MESH CONTROL
        model.mesh.field.add("Distance", 1)
        model.mesh.field.setNumbers(1, "NodesList", [tip_point])
        model.mesh.field.add("Threshold", 2);
        model.mesh.field.setNumber(2, "IField", 1);
        model.mesh.field.setNumber(2, "LcMin", self.lc1)
        model.mesh.field.setNumber(2, "LcMax", self.lc2)
        model.mesh.field.setNumber(2, "DistMin", self.r1)
        model.mesh.field.setNumber(2, "DistMax", Rs)

        model.mesh.field.setAsBackgroundMesh(2)
         
         
        factory.synchronize()
        model.mesh.generate(3)
        gmsh.write(self.file_name + ".msh")
        if use_gui:
            gmsh.fltk.run()
        gmsh.finalize()
        mesh = argiope.mesh.read_msh(self.file_name + ".msh")
        for eset in ["INDENTER_TOP_POINT", 
                     "INDENTER_TIP_POINT",
                     "INDENTER_TOP_SURFACE",
                     "INDENTER_BOTTOM_SURFACE"]:
            mesh.element_set_to_node_set(eset)
            del mesh.elements[("sets", eset, "")]
        mesh.elements = mesh.elements[mesh.space() == 3]
        mesh.node_set_to_surface("INDENTER_BOTTOM_SURFACE")
        self.mesh = mesh  
  
class TransverseFiberSample3D(Sample):
    """
    A Full 3D sample with a tranverse fiber.

    Args:
    * Rf: fiber radius
    * Rs: radius of the outer sphere
    * lc1: characteristic element size at tip.
    * lc2: characteristic element size far from the tip.
    * r1: transition radius for the characteristic length. Below r1 from tip, the elements have a constant size. Above r1, the characteristic length is linerarly increased up to lc2 at Rs distance.
    """
    def __init__(self, Rf = 1., Rs = 10., 
                     lc1 = 0.05, lc2 = 5., 
                     r1 = 0.5, *args, **kwargs):
        self.Rf = Rf
        self.Rs = Rs
        self.r1 = r1
        self.lc1 = lc1
        self.lc2 = lc2
        self.r1 = r1
        super().__init__(**kwargs)
       
    def make_mesh(self, use_gui = False):
        model = gmsh.model
        factory = model.occ

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6);
        model.add("sample")

        lc1 = self.lc1
        lc2 = self.lc2
        Rf = self.Rf
        Rs = self.Rs
        r1 = self.r1

        lx, ly, lz = 2.*Rs, 2.*Rs, 2.*Rs # Box shape
        factory.addCylinder(-2*lx, 0., 0., 4*lx, 0., 0., Rf, 1)
        factory.addBox(-lx/2., -ly/2., -lz, lx, ly, lz, 2)
        factory.intersect([(3,2)], [(3,1)], 
                           removeTool = False, 
                           removeObject = False, 
                           tag = 3)

        factory.cut([(3,2)], [(3,1)], 
                           removeTool = True, 
                           removeObject = True, 
                           tag = 4)
        factory.addSphere(0., 0., 0., Rs, 5)
        factory.intersect([(3,3)], [(3,5)], 6, removeTool = False )
        factory.intersect([(3,4)], [(3,5)], 7, removeTool = True )

        # MODEL ANALYSIS
        factory.synchronize()
        model.addPhysicalGroup(2, [23, 28, 31], 1)
        model.setPhysicalName(2, 1, "SAMPLE_TOP_SURFACE")
        model.addPhysicalGroup(2, [24, 26,27, 29], 2)
        model.setPhysicalName(2, 2, "SAMPLE_BOTTOM_SURFACE")
        model.addPhysicalGroup(3, [6], 1)
        model.setPhysicalName(3, 1, "SAMPLE_FIBER")
        model.addPhysicalGroup(3, [7], 2)
        model.setPhysicalName(3, 2, "SAMPLE_BULK")

        # MESH CONTROL
        top_point = factory.addPoint(0., 0., 0.)
        model.mesh.field.add("Distance", 1)
        model.mesh.field.setNumbers(1, "NodesList", [top_point])
        model.mesh.field.add("Threshold", 2);
        model.mesh.field.setNumber(2, "IField", 1);
        model.mesh.field.setNumber(2, "LcMin", lc1)
        model.mesh.field.setNumber(2, "LcMax", lc2)
        model.mesh.field.setNumber(2, "DistMin", r1)
        model.mesh.field.setNumber(2, "DistMax", Rs)
        model.mesh.field.setAsBackgroundMesh(2)

        # WRITE
        factory.synchronize()
        model.mesh.generate(3)
        #model.mesh.recombine() # CORE DUMP ?
        file_name = self.file_name
        gmsh.write(file_name +".msh")
        if use_gui:
            gmsh.fltk.run()
        gmsh.finalize()
        mesh = argiope.mesh.read_msh(file_name + ".msh")
        for eset in ["SAMPLE_TOP_SURFACE", 
                     "SAMPLE_BOTTOM_SURFACE"]:
            mesh.element_set_to_node_set(eset)
            del mesh.elements[("sets", eset, "")]
        mesh.elements = mesh.elements[mesh.space() == 3]
        mesh.node_set_to_surface("SAMPLE_TOP_SURFACE")
        self.mesh = mesh    

  
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
                         friction = 0.,
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
        MATERIALS = "\n".join([m.write_inp() for m in materials]),
        FRICTION = friction )
  if path == None:            
    return pattern
  else:
    open(path, "w").write(pattern)  
