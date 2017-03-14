classdef pointlander2d
    properties
        s0;
        st;
        nnodes;
        Tm;
        Isp;
    end
    methods
        function self = pointlander2d(s0,st,Tm,Isp,nnodes)
            self.s0 = s0;
            self.st = st;
            self.Isp = Isp;
            