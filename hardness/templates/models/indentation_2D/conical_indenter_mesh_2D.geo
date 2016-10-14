// SETTINGS
r1 =  $r1;
r2 =  $r2;
lc1 = $lc1;
lc2 = $lc2;


Point(1) = {0. ,  0. , 0., lc1};
Point(2) = {$x2,  $y2, 0., lc2};
Point(3) = {0. ,  $y3, 0., lc2};

Line(1)      = {1,2};
Circle(2)    = {2,1,3};
Line(3)      = {3,1};
Line Loop(1) = {1,2,3};
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
Physical Line("SURFACE") = {1};
Physical Line("BOTTOM") = {2};
Physical Line("AXIS") = {3};
Physical Surface("ALL_ELEMENTS") = {1};


