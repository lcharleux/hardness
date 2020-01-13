import hardness as hd

workdir = "./workdir/"

mesh = hd.models.sample_mesh_2D("gmsh", 
                             workdir, 
                             lx = 1., 
                             ly = 1., 
                             r1 = 2., 
                             r2 = 500., 
                             Nx = 64, 
                             Ny = 32,
                             Nr = 32,
                             Nt = 32,  
                             algorithm = "delquad")
