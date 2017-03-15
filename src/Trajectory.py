from numpy import *

''' -----------
Dynamical Model
----------- '''
class Dynamical_Model(object):
    def __init__(self, si, st, slb, sub, clb, cub, tlb, tub):
        self.si  , self.st    = array(si) , array(st)
        self.slb , self.sub   = array(slb), array(sub)
        self.clb , self.cub   = array(clb), array(cub)
        self.tlb , self.tub   = array(tlb), array(tub)
        self.sdim, self.cdim  = len(slb)  , len(clb)
    def __repr__(self):
        n, t = "\n", "\t"
        lb, ub, dim = "Lower Bound: ", "Upper Bound: ", "Dimensions: "
        s = "State" + n
        s += t + dim + str(self.sdim) + n
        s += t + "Initial: " + str(self.si) + n
        s += t + "Target: " + str(self.st) + n
        s += t + lb + str(self.slb) + n
        s += t + ub + str(self.sub) + n
        s += "Control" + n
        s += t + dim + str(self.cdim) + n
        s += t + lb + str(self.clb) + n
        s += t + ub + str(self.cub) + n
        s += "Time" + n
        s += t + lb + str(self.tlb) + n
        s += t + ub + str(self.tub) + n
        return s
    def EOM_State(self, state, control):
        return None
    def EOM_Jac(self, state, control):
        return None
    def EOM_Fullstate(self, fullstate, control):
        return None
    def EOM_Fullstate_Jac(self, fullstate, control):
        return None

''' ---
Landers
--- '''
class Point_Lander(Dynamical_Model):
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
        Dynamical_Model.__init__(
            self,
            si,
            st,
            [-500, 0, -200, -200, 0],
            [500, 1000, 200, 200, 10000],
            [0, -1, -1],
            [1, 1, 1],
            1,
            1000
        )
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

if __name__ == "__main__":
    Apollo = Point_Lander()
    print Apollo.EOM_Fullstate(Apollo.sub, Apollo.cub)
