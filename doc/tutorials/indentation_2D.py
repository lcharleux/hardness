import hardness as hd
import os


workdir = "./workdir/"
simName = "indentation_2D"

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
                                   
hd.models.indentation_2D_input(sample_mesh   = sample_mesh,
                            indenter_mesh = indenter_mesh, 
                            path = workdir + simName + ".inp")
                                      
hd.postproc.indentation_abqpostproc(
        workdir     =  workdir, 
        path        = simName + "_abqpostproc.py", 
        odbPath     = simName + ".odb", 
        histPath    = simName + "_hist", 
        contactPath = simName + "_contact", 
        fieldPath   = simName + "_fields")
        
hd.postproc.indentation_pypostproc(
        workdir     =  workdir, 
        path        = simName + "_pypostproc.py", 
        histPath    = simName + "_hist", 
        contactPath = simName + "_contact", 
        fieldPath   = simName + "_fields")                                              
