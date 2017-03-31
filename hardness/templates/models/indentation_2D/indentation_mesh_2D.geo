// SETTINGS
lx = $lx;
ly = $ly;
r1 = $r1;
r2 = $r2;
Nx = $Nx + 1;
Ny = $Ny + 1;
Nr = $Nr + 1;
Nt = $Nt + 1;
lc0 = ly/Ny;
q1 = (r2/r1)^(1./Nr); 
lcx = lx / Nx; 
lcy = ly / Ny;
lcxy = (lcx + lcy) / 2.;
lcs = 3.14159265 / 2. * r1 / Nt; 
diag = (lx^2 + ly^2)^.5;

Point(1) = {0.,  0., 0., lcxy};
Point(2) = {0,  -ly, 0., lcx};
Point(3) = {lx, -ly, 0., lcxy};
Point(4) = {lx,  0., 0., lcy};
Point(5) = {0., -r1, 0., lcs};
Point(6) = {r1,  0., 0., lcs};
Point(7) = {0., -r2, 0., lcs};
Point(8) = {r2,  0., 0., lcs};


// Center square
Line(1)  = {1,2};
Line(2)  = {2,3};
Line(3)  = {3,4};
Line(4)  = {4,1};
Transfinite Line {2,4} = Nx;
Transfinite Line {1,3} = Ny;
Line Loop(1)           = {1,2,3,4};
Plane Surface(1) = {1};
Transfinite Surface {1};


// Shell 1
Line(5)   = {2,5};
Circle(6) = {5,1,6};
Line(7)   = {4,6};
Transfinite Line {6} = Nt;
Line Loop(2) = {5,6,-7,-3,-2};
Plane Surface(2) = {2};

// Shell 2
Line(8)   = {5,7};
Circle(9) = {7,1,8};
Line(10)   = {6,8};
Line Loop(3) = {8,9,-10,-6};
Transfinite Line {9} = Nt;
Transfinite Line {8,10} = Nr Using Progression q1;
Plane Surface(3) = {3};
Transfinite Surface {3};




Recombine Surface {1,2,3};
Physical Line("SURFACE") = {4,7};
Physical Line("BOTTOM") = {9};
Physical Line("AXIS") = {1,5,8};
Physical Surface("CORE") = {1};
Physical Surface("SHELL1") = {2};
Physical Surface("SHELL2") = {3};


