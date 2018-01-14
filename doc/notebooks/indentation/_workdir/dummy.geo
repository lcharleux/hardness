// SETTINGS
r1 =  1.0;
r2 =  3.0;
lc1 = 0.05;
lc2 = 0.5;

Point(1) = {0. ,  0. , 0., lc1};
Point(2) = {0.337095258423,  0.058529455188, 0., lc2};
Point(3) = {3.76588217925,  1.28621291222, 0., lc2};
Point(4) = {0. ,  3.93783187853, 0., lc2};
Point(5) = {0. ,  1.0, 0., lc2};
Point(6) = {0. ,  -0.0621681214675, 0., lc2};


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


