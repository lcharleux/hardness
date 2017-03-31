import hardness as hd
import argiope as ag
import os, subprocess, time, local_settings

#-------------------------------------------------------------------------------
# 2D INDENTATION WITH HARDNESS + ARGIOPE + ABAQUS
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# USEFUL FUNCTIONS
def create_dir(path):
  try:
    os.mkdir(path)
  except:
    pass
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# SETTINGS
workdir   = "workdir/"
outputdir = "outputs/"
label   = "indentation_2D"
#-------------------------------------------------------------------------------


create_dir(workdir)
create_dir(workdir + outputdir)

#-------------------------------------------------------------------------------

                                                                  
                                                                  
                                                                  
class Indentation2D(ag.models.Model):
  """
  2D indentation class.
  """
    
  def write_input(self):
    """
    Writes the input file in the chosen format.
    """
    hd.models.indentation_2D_input(sample_mesh   = self.meshes["sample"],
                                   indenter_mesh = self.meshes["indenter"],
                                   steps = self.steps,
                                   solver = self.solver,
                                   path = "{0}/{1}.inp".format(self.workdir,
                                                               self.label))
                                   
                                   
    
  def write_postproc(self):
    """
    Writes the prosproc scripts
    """
    if self.solver == "abaqus":
      hd.postproc.indentation_abqpostproc(
          path = "{0}/{1}_abqpp.py".format(
              self.workdir,
              self.label),
          label = self.label,    
          solver= self.solver)
    


#-------------------------------------------------------------------------------
# MESH DEFINITIONS
meshes = {
    "sample" : hd.models.sample_mesh_2D("gmsh", 
                                   workdir, 
                                   lx = 1., 
                                   ly = 1., 
                                   r1 = 2., 
                                   r2 = 100., 
                                   Nx = 32, 
                                   Ny = 16, 
                                   lc1 = 0.2, 
                                   lc2 = 20.),
                                   
    "indenter" : hd.models.spheroconical_indenter_mesh_2D("gmsh", 
                                   workdir, 
                                   R = 1.,
                                   psi= 30., 
                                   r1 = 1., 
                                   r2 = 100., 
                                   r3 = 100., 
                                   lc1 = 0.1, 
                                   lc2 = 20.)}
                                   
"""
sample_mesh.save(h5path = workdir + outputdir + simName + "_sample_mesh.h5")
indenter_mesh.save(h5path = workdir + outputdir + simName + "_indenter_mesh.h5")  
"""   
#-------------------------------------------------------------------------------
# STEP DEFINTIONS
steps = [
        hd.models.indentation_2D_step_input(name = "LOADING1",
                                            control_type = "disp", 
                                            duration = 1.,
                                            kind = "adaptative",  
                                            nframes = 50,
                                            controlled_value = -0.1),
        hd.models.indentation_2D_step_input(name = "UNLOADING1",
                                            control_type = "force", 
                                            duration = 1.,
                                            kind = "adaptative",  
                                            nframes = 50,
                                            controlled_value = 0.),
        hd.models.indentation_2D_step_input(name = "RELOADING1",
                                            control_type = "disp", 
                                            duration = 1.,
                                            kind = "adaptative",  
                                            nframes = 50,
                                            controlled_value = -0.1),
        hd.models.indentation_2D_step_input(name = "LOADING2",
                                            control_type = "disp", 
                                            duration = 1.,
                                            kind = "adaptative",  
                                            nframes = 50,
                                            controlled_value = -0.2),                                    
        hd.models.indentation_2D_step_input(name = "UNLOADING2",
                                            control_type = "force",
                                            kind = "adaptative", 
                                            duration = 1., 
                                            nframes = 50,
                                            controlled_value = 0.)
        ]                                                                                                  
#-------------------------------------------------------------------------------
"""                                  
hd.models.indentation_2D_input(sample_mesh   = sample_mesh,
                            indenter_mesh = indenter_mesh,
                            steps = steps ,
                            path = workdir + simName + ".inp")
                                     
hd.postproc.indentation_abqpostproc(path = workdir + simName + "_abqpp.py")        
hd.postproc.indentation_pypostproc(path = workdir + simName + "_pypp.py") 
"""
#-------------------------------------------------------------------------------
model = Indentation2D(label = label, 
                      meshes = meshes, 
                      steps = steps, 
                      materials = None, 
                      solver = "abaqus", 
                      solver_path = local_settings.ABAQUS_PATH,
                      workdir = workdir,
                      verbose = True)

model.write_input()
model.run_simulation()
model.write_postproc()
model.run_postproc()
