// SETTINGS
r1 =  1.0;
r2 =  3.0;
lc1 = 0.03;
lc2 = 0.5;

Point(1) = {0. ,  0. , 0., lc1};
Point(2) = {0.866025403784,  0.5, 0., lc2};
Point(3) = {1.5,  1.59807621135, 0., lc2};
Point(4) = {0. ,  2.0, 0., lc2};
Point(5) = {0. ,  1.0, 0., lc2};
Point(6) = {0. ,  -1.0, 0., lc2};


Circle(1) = {1,5,2};
Line(2) = {2,3};
Circle(3) = {3,6,4};
Line(4) = {4,1};
Line Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};

Field[1] = Attractor;
Field[1].NodesList = {1};
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc1;
Field[2].LcMax = lc2;
Field[2].DistMin = r1;
Field[2].DistMax = r2;
Background Field = 2;


Recombine Surface {1};
Physical Line("SURFACE") = {1,2};
Physical Line("BOTTOM") = {3};
Physical Line("AXIS") = {4};
Physical Surface("ALL_ELEMENTS") = {1};


