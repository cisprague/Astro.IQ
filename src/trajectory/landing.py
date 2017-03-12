from numpy import *
from PyGMO import *

class PointLander(object):
    def __init__(self,
        si  = [0, 1000, 20, -5, 10000],
        st  = [0, 0, 0, 0, 9000],
        Isp = 311,
        g   = 1.6229,
        a   = 0,
        pin = False,
        T   = 44000
    ):
        self.si  = array(si, float)
        self.st  = array(st, float)
        self.Isp = float(Isp)
        self.g   = float(g)
        self.a   = float(a)
        self.g0  = float(9.81)
        self.pin = bool(pin)
        self.T   = float(T)
        return None
    def Indirect_Objective(self, decision):
        return (1., )
    def Indirect_Constraints(self, decision):
        traj, cont, t = self.Shoot(decision)
        x, y, vx, vy, m, lx, ly, lvx, lvy, lm = traj[-1]
        u, st, ct, S, H = cont[-1]
        xt, yt, vxt, vyt, mt = self.st
        ceq = empty(5)
        if self.pin:
            ceq[0] = x - xt
        ceq[1] = y - yt
        ceq[2] = vx - vxt
        ceq[3] = vy - vyt
        ceq[4] = lm
        ceq[5] = H
        return ceq
    def Shoot(self, decision):
        tf, lx, ly, lvx, lvy, lm = decision
        # TODO: Integrate!
        return None # for now...zzz...zzz
    def EOM_State(self, jac=True):
        x, y, vx, vy, m = self.state
        u, st, ct, S, H = self.control
        x0 = self.T*u/m**2
        ds = array([
            vx,
            vy,
            st*x0,
            ct*x0 - self.g0,
            -self.T*u/(self.Isp*self.g0)
        ], float)
        if jac is True:
            dds = array([
                [0, 0, 1, 0,      0],
                [0, 0, 0, 1,      0],
                [0, 0, 0, 0, -st*x0],
                [0, 0, 0, 0, -ct*x0],
                [0, 0, 0, 0,      0]
            ], float)
            return ds, jac
        else:
            return ds
    def EOM_Costate(self, jac=True):
        x, y, vx, vy, m      = self.state
        lx, ly, lvx, lvy, lm = self.costate
        u, st, ct, S, H      = self.control
        x0 = self.T*u/m**2
        dl = array([
            0,
            0,
            -lx,
            -ly,
            ct*lvy*x0 + st*lvx*x0
        ], float)
        if jac is True:
            ddl = array([
                [ 0,  0,     0,     0, 0],
                [ 0,  0,     0,     0, 0],
                [-1,  0,     0,     0, 0],
                [ 0, -1,     0,     0, 0],
                [ 0,  0, st*x0, ct*x0, 0]
            ], float)
            return dl, ddl
        else:
            return dl

if __name__ == "__main__":
    sc = PointLander()
    print empty(5)
