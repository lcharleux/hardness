import hardness as hd
import os

def create_dir(path):
  try:
    os.mkdir(path)
  except:
    pass


workdir   = "workdir/"
outputdir = "outputs/"
simName   = "indentation_2D"

create_dir(workdir)
create_dir(workdir + outputdir)


sample_mesh = hd.models.sample_mesh_2D("gmsh", 
                                   workdir, 
                                   lx = 1., 
                                   ly = 1., 
                                   r1 = 2., 
                                   r2 = 100., 
                                   Nx = 32, 
                                   Ny = 16, 
                                   lc1 = 0.2, 
                                   lc2 = 20.)
                                   
indenter_mesh = hd.models.spheroconical_indenter_mesh_2D("gmsh", 
                                   workdir, 
                                   R = 1.,
                                   psi= 30., 
                                   r1 = 1., 
                                   r2 = 100., 
                                   r3 = 100., 
                                   lc1 = 0.1, 
                                   lc2 = 20.)

sample_mesh.save(h5path = workdir + outputdir + simName + "_sample_mesh.h5")
indenter_mesh.save(h5path = workdir + outputdir + simName + "_indenter_mesh.h5")     
                                 
hd.models.indentation_2D_input(sample_mesh   = sample_mesh,
                            indenter_mesh = indenter_mesh, 
                            path = workdir + simName + ".inp")
                                      
hd.postproc.indentation_abqpostproc(
        workdir     =  workdir, 
        odbPath     = simName)
        
hd.postproc.indentation_pypostproc(
        workdir     =  workdir, 
        odbPath        = simName)                                              
