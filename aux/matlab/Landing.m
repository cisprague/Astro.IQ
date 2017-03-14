Apollo = Lander([0;1000;-5;0;10000],[0;0;0;0;0],44000,311*9.81,1.6229,0)
dec = [100;1;1;1;1;1];
Apollo.Constraints(dec);
s = Apollo.s0;
l = [0.5;0.5;0.5;0.5;0.5];
fs = [s;l];
dec = Apollo.Optimise()