Mesh.MshFileVersion = 2.2;
lc=1.5;
Point(1) = { 5.515664, 0.0, 3.3, lc };
Point(2) = { 5.515664, 0.0, 0.0, lc };
Point(3) = { 6.213329, 4.019065, 0.0, lc };
Point(4) = { 6.213329, 4.019065, 3.3, lc };
Point(5) = { 5.87996, 2.098616, 2.5, lc };
Point(6) = { 5.87996, 2.098616, 1.02, lc };
Point(7) = { 5.666172, 0.867034, 1.02, lc };
Point(8) = { 5.666172, 0.867034, 2.5, lc };
Point(9) = { 2.6, 0.0, 0.0, lc };
Point(10) = { 1.905664, 0.0, 0.0, lc };
Point(11) = { 1.905664, 3.84, 0.0, lc };
Point(12) = { 4.785664, 3.84, 0.0, lc };
Point(13) = { 4.785664, 0.0, 0.0, lc };
Point(14) = { 0.5, 0.0, 0.0, lc };
Point(15) = { 0.0, 0.0, 0.0, lc };
Point(16) = { 0.0, 0.57, 0.0, lc };
Point(17) = { 0.0, 1.77, 0.0, lc };
Point(18) = { 0.0, 5.09763, 0.0, lc };
Point(19) = { 1.55672, 4.827401, 0.0, lc };
Point(20) = { 2.739038, 4.622163, 0.0, lc };
Point(21) = { 4.646757, 4.291005, 0.0, lc };
Point(22) = { 5.829075, 4.085767, 0.0, lc };
Point(23) = { 1.55672, 4.827401, 2.7, lc };
Point(24) = { 0.0, 5.09763, 3.3, lc };
Point(25) = { 5.829075, 4.085767, 2.7, lc };
Point(26) = { 4.646757, 4.291005, 2.7, lc };
Point(27) = { 2.739038, 4.622163, 2.7, lc };
Point(28) = { 0.0, 0.0, 3.3, lc };
Point(29) = { 0.5, 0.0, 2.165, lc };
Point(30) = { 2.6, 0.0, 2.165, lc };
Point(31) = { 0.0, 0.57, 2.7, lc };
Point(32) = { 0.0, 1.77, 2.7, lc };
Line(1) = { 1, 2 };
Line(2) = { 2, 3 };
Line(3) = { 3, 4 };
Line(4) = { 1, 4 };
Line(5) = { 5, 6 };
Line(6) = { 6, 7 };
Line(7) = { 7, 8 };
Line(8) = { 5, 8 };
Line(9) = { 9, 10 };
Line(10) = { 10, 11 };
Line(11) = { 11, 12 };
Line(12) = { 12, 13 };
Line(13) = { 9, 13 };
Line(14) = { 2, 13 };
Line(15) = { 10, 14 };
Line(16) = { 14, 15 };
Line(17) = { 15, 16 };
Line(18) = { 16, 17 };
Line(19) = { 17, 18 };
Line(20) = { 18, 19 };
Line(21) = { 19, 20 };
Line(22) = { 20, 21 };
Line(23) = { 21, 22 };
Line(24) = { 3, 22 };
Line(25) = { 19, 23 };
Line(26) = { 18, 24 };
Line(27) = { 4, 24 };
Line(28) = { 22, 25 };
Line(29) = { 25, 26 };
Line(30) = { 21, 26 };
Line(31) = { 20, 27 };
Line(32) = { 23, 27 };
Line(33) = { 1, 28 };
Line(34) = { 15, 28 };
Line(35) = { 14, 29 };
Line(36) = { 29, 30 };
Line(37) = { 9, 30 };
Line(38) = { 31, 32 };
Line(39) = { 17, 32 };
Line(40) = { 16, 31 };
Line(41) = { 24, 28 };
Line Loop(1) = { 1, 2, 3, -4 };
Line Loop(2) = { 5, 6, 7, -8 };
Line Loop(3) = { 9, 10, 11, 12, -13 };
Line Loop(4) = { 14, -12, -11, -10, 15, 16, 17, 18, 19, 20, 21, 22, 23, -24, -2 };
Line Loop(5) = { -25, -20, 26, -27, -3, 24, 28, 29, -30, -22, 31, -32 };
Line Loop(6) = { -21, 25, 32, -31 };
Line Loop(7) = { 13, -14, -1, 33, -34, -16, 35, 36, -37 };
Line Loop(8) = { 38, -39, -18, 40 };
Line Loop(9) = { -5, 8, -7, -6 };
Line Loop(10) = { 37, -36, -35, -15, -9 };
Line Loop(11) = { 27, 41, -33, 4 };
Line Loop(12) = { -26, -19, 39, -38, -40, -17, 34, -41 };
Line Loop(13) = { -23, 30, -29, -28 };
Plane Surface(1) = { 1, 2 };
Plane Surface(2) = { 3 };
Plane Surface(3) = { 4 };
Plane Surface(4) = { 5 };
Plane Surface(5) = { 6 };
Plane Surface(6) = { 7 };
Plane Surface(7) = { 8 };
Plane Surface(8) = { 9 };
Plane Surface(9) = { 10 };
Plane Surface(10) = { 11 };
Plane Surface(11) = { 12 };
Plane Surface(12) = { 13 };
Surface Loop(1) = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
Physical Surface(11) = { 1, 3, 4, 6, 8, 9, 10, 11 }; //hard surface
Physical Surface(13) = { 2 }; //carpet on the floor
Physical Surface(14) = { 5, 7, 12 }; //panel on the wall
Volume( 1 ) = { 1 };
Physical Volume(1) = { 1 };
//Physical Line ("default") = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41};
Mesh.Algorithm = 1;
Mesh.Algorithm3D = 1; // Delaunay3D, works for boundary layer insertion.
Mesh.Optimize = 1; // Gmsh smoother, works with boundary layers (netgen version does not).
Mesh.CharacteristicLengthFromPoints = 1;
// Recombine Surface "*";

