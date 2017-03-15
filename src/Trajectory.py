from numpy import *

class PointLander(object):
    def __init__(
        self,
        si  = [10, 1000, 20, -5, 10000],
        st  = [0, 0, 0, 0, 9000],
        Isp = 311,
        g   = 1.6229,
        T   = 44000):
        # Problem parameters
        self.Isp  = float(Isp)
        self.g    = float(g)
        self.T    = float(T)
        self.g0   = float(9.802)
        # For optimisation
        self.si   = array(si, float)
        self.st   = array(st, float)
        self.sdim = 5
        self.cdim = 3
        self.slb  = [-500, 0, -200, -200, 0]
        self.sub  = [500, 1000, 200, 200, 10000]
        self.clb  = [0, -1, -1]
        self.cub  = [1, 1, 1]
        self.tlb  = [1]
        self.tub  = [1000]
    def EOM_State(self, state, control):
        x, y, vx, vy, m = state
        u, st, ct       = control
        x0 = self.T*u/m
        return array([
            vx,
            vy,
            st*x0,
            ct*x0 - self.g,
            -self.T*u/(self.Isp*self.g0)
        ], float)
        return ds
    def EOM_State_Jac(self, state, control):
        x, y, vx, vy, m = state
        u, st, ct       = control
        x0              = self.T*u/m**2
        return array([
            [0, 0, 1, 0,        0],
            [0, 0, 0, 1,        0],
            [0, 0, 0, 0, -st*x0/m],
            [0, 0, 0, 0, -ct*x0/m],
            [0, 0, 0, 0,        0]
        ], float)
