// SETTINGS
Rf = $Rf;
ly1 = $ly1;
ly2 = $ly2;
Nx = $Nx + 1;
Ny = $Ny + 1;
Nr = $Nr + 1;
Nt = $Nt;

lcx = Rf / Nx; 
lcy = ly1 / Ny;
q1 = (ly2/ly1)^(1./Nr); 


Point(1) = {0.,  0.,  0., lcy};
Point(2) = {0.,  -ly1, 0., lcy};
Point(3) = {Rf,  -ly1, 0., lcy};
Point(4) = {Rf,  0,   0., lcy};
Point(5) = {0,  -ly2, 0., lcy};
Point(6) = {Rf,  -ly2,   0., lcy};
Point(7) = {Rf + ly1, 0,   0., lcy};
Point(8) = {Rf + ly2, 0,   0., lcy};


Line(1)  = {1,2};
Line(2)  = {2,3};
Line(3)  = {3,4};
Line(4)  = {4,1};
Line(5)  = {2,5};
Line(6)  = {5,6};
Line(7)  = {6,3};
Line(8)  = {4,7};
Line(9)  = {7,8};
Circle(10) = {3,4,7};
Circle(11) = {6,4,8};


Line Loop(1)           = {1,2,3,4};
Plane Surface(1) = {1};
Transfinite Surface {1};

Line Loop(2)           = {5,6,7,-2};
Plane Surface(2) = {2};
//Transfinite Surface {2};

Line Loop(3)           = {-3, 10, -8};
Plane Surface(3) = {3};

Line Loop(4)           = {-7, 11, -9, -10};
Plane Surface(4) = {4};

Transfinite Line {4,2,6} = Nx;
Transfinite Line {1,3} = Ny;
Transfinite Line {10,11} = Nt;
Transfinite Line {5,-7, 9} = Nr Using Progression q1;
Transfinite Surface {1, 2, 4};


Recombine Surface {1, 2, 3, 4};
Physical Line("SURFACE") = {4,9};
Physical Line("BOTTOM") = {6, 11};
Physical Line("AXIS") = {1,5};
Physical Surface("FIBRE") = {1, 2};
Physical Surface("MATRIX") = {3, 4};


