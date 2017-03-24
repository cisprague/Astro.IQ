'''
Astro.IQ - Trajectory
Christopher Iliffe Sprague
christopher.iliffe.sprague@gmail.com
https://cisprague.github.io/Astro.IQ
'''

''' --------
Dependencies
-------- '''
from numpy import *
from scipy.integrate import odeint
from numba import jit, jitclass, float32

''' -----------
Dynamical Model
----------- '''
class Dynamical_Model(object):
    def __init__(self, si, st, slb, sub, clb, cub, tlb, tub, silb, siub):
        # Initial and target state
        self.si  , self.st   = array(si , float), array(st , float)
        # Lower and upper state bounds
        self.slb , self.sub  = array(slb, float), array(sub, float)
        # Lower and upper control bounds
        self.clb , self.cub  = array(clb, float), array(cub, float)
        # Lower and upper time bounds
        self.tlb , self.tub  = float(tlb), float(tub)
        # State and control space dimensions
        self.sdim, self.cdim = len(slb)  , len(clb)
        # Numerical integration
        self.Propagate       = Propagate(self)
        # Lower and upper initial state bounds
        self.silb, self.siub = array(silb, float), array(siub, float)
    def __repr__(self):
        n, t = "\n", "\t"
        lb, ub, dim = "Lower Bound: ", "Upper Bound: ", "Dimensions: "
        s = "State"                           + n
        s += t + dim         + str(self.sdim) + n
        s += t + "Initial: " + str(self.si)   + n
        s += t + "Target: "  + str(self.st)   + n
        s += t + lb          + str(self.slb)  + n
        s += t + ub          + str(self.sub)  + n
        s += "Control"                        + n
        s += t + dim         + str(self.cdim) + n
        s += t + lb          + str(self.clb)  + n
        s += t + ub          + str(self.cub)  + n
        s += "Time"                           + n
        s += t + lb          + str(self.tlb)  + n
        s += t + ub          + str(self.tub)  + n
        return s

''' ---
Landers
--- '''

class Point_Lander(Dynamical_Model):
    def __init__(
        self,
        si  = [10, 1000, 20, -5, 9500],
        st  = [0, 0, 0, 0, 8000],
        Isp = 311,
        g   = 1.6229,
        T   = 44000,
        a   = 0
        ):
        # Problem parameters
        self.Isp  = float(Isp)   # Specific impulse [s]
        self.g    = float(g)     # Environment's gravity [m/s^2]
        self.T    = float(T)     # Maximum thrust [N]
        self.g0   = float(9.802) # Earth's sea-level gravity [m/s^2]
        self.a    = float(a)     # Homotopy parametre
        # For optimisation
        Dynamical_Model.__init__(
            self,
            si,
            st,
            [-10000, 0, -500, -500, 0],
            [10000, 2000, 500, 500, 10000],
            [0, -1, -1],
            [1, 1, 1],
            1,
            200,
            [-400, 500, -150, -200, 8000],
            [400, 1000, 150, 2, 9800]
        )
    def EOM_State(self, state, control):
        x, y, vx, vy, m = state
        u, ux, uy       = control
        x0 = self.T*u/m
        return array([
            vx,
            vy,
            ux*x0,
            uy*x0 - self.g,
            -self.T*u/(self.Isp*self.g0)
        ], float)
    def EOM_State_Jac(self, state, control):
        x, y, vx, vy, m = state
        u, ux, uy       = control
        x0              = self.T*u/m**2
        return array([
            [0, 0, 1, 0,        0],
            [0, 0, 0, 1,        0],
            [0, 0, 0, 0, -ux*x0/m],
            [0, 0, 0, 0, -uy*x0/m],
            [0, 0, 0, 0,        0]
        ], float)
    def EOM_Fullstate(self, fullstate, control):
        x, y, vx, vy, m, lx, ly, lvx, lvy, lm = fullstate
        u, ux, uy     = control
        T, Isp, g0, g = self.T, self.Isp, self.g0, self.g
        x0            = T*u/m
        x1            = T*u/m**2
        return array([
            vx,
            vy,
            ux*x0,
            uy*x0 - g,
            -T*u/(Isp*g0),
            0,
            0,
            -lx,
            -ly,
            uy*lvy*x1 + lvx*ux*x1
        ], float)
    def EOM_Fullstate_Jac(self, fullstate, control):
        x, y, vx, vy, m, lx, ly, lvx, lvy, lm = fullstate
        u, ux, uy = control
        T         = self.T
        x0        = T*u/m**2
        x1        = ux*x0
        x2        = uy*x0
        x3        = 2*T*u/m**3
        return array([
            [0, 0, 1, 0,                      0,  0,  0,  0,  0, 0],
            [0, 0, 0, 1,                      0,  0,  0,  0,  0, 0],
            [0, 0, 0, 0,                    -x1,  0,  0,  0,  0, 0],
            [0, 0, 0, 0,                    -x2,  0,  0,  0,  0, 0],
            [0, 0, 0, 0,                      0,  0,  0,  0,  0, 0],
            [0, 0, 0, 0,                      0,  0,  0,  0,  0, 0],
            [0, 0, 0, 0,                      0,  0,  0,  0,  0, 0],
            [0, 0, 0, 0,                      0, -1,  0,  0,  0, 0],
            [0, 0, 0, 0,                      0,  0, -1,  0,  0, 0],
            [0, 0, 0, 0, -uy*lvy*x3 - lvx*ux*x3,  0,  0, x1, x2, 0]
        ], float)
    def Hamiltonian(self, fullstate, control):
        x, y, vx, vy, m, lx, ly, lvx, lvy, lm = fullstate
        u, ux, uy = control
        T, Isp, g0, g, a = self.T, self.Isp, self.g0, self.g, self.a
        x0 = T*u/m
        x1 = 1/(Isp*g0)
        H  = -T*lm*u*x1 + lvx*ux*x0 + lvy*(uy*x0 - g) + lx*vx + ly*vy
        H += x1*(T**2*u**2*(-a + 1) + a*(T*u)) # The Lagrangian
        return H
    def Pontryagin(self, fullstate):
        x, y, vx, vy, m, lx, ly, lvx, lvy, lm = fullstate
        lv = sqrt(abs(lvx)**2 + abs(lvy)**2)
        ux = -lvx/lv
        uy = -lvy/lv
        # Switching function
        S = self.a - lv*self.Isp*self.g0/m - lm
        if self.a == 1.:
            if S >= 0.: u = 0.
            elif S < 0.: u = 1.
        else:
            u = -S/(2*self.T*(1-self.a))
            u = min(u, 1.)
            u = max(u, 0.)
        return u, ux, uy
class Rocket(Dynamical_Model):
    def __init__(self,
        si = [0, 1000, 20, -5, 0, 0, 10000],
        st = [0, 0, 0, 0, 0, 0, 9000],
        Tmax=44000, Isp=311, a=0, g=9.81, charlen=300):
        # Model parametres
        self.c1 = float(Tmax)
        self.c2 = float(Isp*9.81)
        self.c3 = float(charlen)
        self.g  = g
        self.a  = a
        # Problem parametres
        Dynamical_Model.__init__(
            self,
            si,
            st,
            [-10000, 0, -500, -500, 0, -10, 1],
            [10000, 2000, 500, 500, pi, 10, 10000],
            [-1e-5, -1, -1],
            [1, 1, 1],
            1,
            100,
            [-200, 500, -200, -200, 8000],
            [200, 1000, 200, -10, 9900]
        )
    @staticmethod
    def EOM_State(state, control, c1, c2, c3, g):
        x, y, vx, vy, theta, omega, m = state
        u, ut1, ut2 = control
        x0 = c1*u/m
        x1 = ut1*x0
        return array([
            [                                      vx],
            [                                      vy],
            [                                      x1],
            [                                 -g + x1],
            [                                   omega],
            [-x0*(ut1*sin(theta) + ut1*cos(theta))/c3],
            [                                -c1*u/c2]
        ], float)
    @staticmethod
    def EOM_State_Jac(state, control, c1, c2, c3, g):
        x0 = c1*u/m**2
        x1 = -ut1*x0
        x2 = 1/c3
        x3 = ut1*cos(theta)
        x4 = ut1*sin(theta)
        return array([
            [0, 0, 1, 0,                    0, 0,               0],
            [0, 0, 0, 1,                    0, 0,               0],
            [0, 0, 0, 0,                    0, 0,              x1],
            [0, 0, 0, 0,                    0, 0,              x1],
            [0, 0, 0, 0,                    0, 1,               0],
            [0, 0, 0, 0, -c1*u*x2*(x3 - x4)/m, 0, x0*x2*(x3 + x4)],
            [0, 0, 0, 0,                    0, 0,               0]
        ], float)
    @staticmethod
    def EOM_Fullstate(fullstate, control, c1, c2, c3, g):
        x,y,vx,vy,theta,omega,m,lx,ly,lvx,lvy,ltheta,lomega,lm = fullstate
        u, ut1, ut2 = control
        x0 = c1*u/m
        x1 = 1/c3
        x2 = cos(theta)
        x3 = sin(theta)
        x4 = x1*(ut1*x2 + ut2*x3)
        x5 = m**(-2)
        x6 = c1*u*x5
        return array([
            [                                          vx],
            [                                          vy],
            [                                      ut1*x0],
            [                                 -g + ut2*x0],
            [                                       omega],
            [                                      -x0*x4],
            [                                    -c1*u/c2],
            [                                           0],
            [                                           0],
            [                                         -lx],
            [                                         -ly],
            [             lomega*x0*x1*(-ut1*x3 + ut2*x2)],
            [                                     -ltheta],
            [-c1*lomega*u*x4*x5 + lvx*ut1*x6 + lvy*ut2*x6]
        ], float)
    @staticmethod
    def EOM_Fullstate_Jac(fullstate, control, c1, c2, c3, g):
        x,y,vx,vy,theta,omega,m,lx,ly,lvx,lvy,ltheta,lomega,lm = fullstate
        u, ut1, ut2 = control
        x0  = m**(-2)
        x1  = c1*u*x0
        x2  = ut1*x1
        x3  = ut2*x1
        x4  = c1*u/m
        x5  = 1/c3
        x6  = cos(theta)
        x7  = sin(theta)
        x8  = x5*(-ut1*x7 + ut2*x6)
        x9  = x4*x8
        x10 = ut1*x6
        x11 = ut2*x7
        x12 = x5*(x10 + x11)
        x13 = x1*x12
        x14 = -c1*lomega*u*x0*x8
        x15 = m**(-3)
        x16 = 2*c1*u*x15
        return array([
            [0, 0, 1, 0,                         0, 0,                                                 0,  0,  0,  0,  0,  0,    0, 0],
            [0, 0, 0, 1,                         0, 0,                                                 0,  0,  0,  0,  0,  0,    0, 0],
            [0, 0, 0, 0,                         0, 0,                                               -x2,  0,  0,  0,  0,  0,    0, 0],
            [0, 0, 0, 0,                         0, 0,                                               -x3,  0,  0,  0,  0,  0,    0, 0],
            [0, 0, 0, 0,                         0, 1,                                                 0,  0,  0,  0,  0,  0,    0, 0],
            [0, 0, 0, 0,                       -x9, 0,                                               x13,  0,  0,  0,  0,  0,    0, 0],
            [0, 0, 0, 0,                         0, 0,                                                 0,  0,  0,  0,  0,  0,    0, 0],
            [0, 0, 0, 0,                         0, 0,                                                 0,  0,  0,  0,  0,  0,    0, 0],
            [0, 0, 0, 0,                         0, 0,                                                 0,  0,  0,  0,  0,  0,    0, 0],
            [0, 0, 0, 0,                         0, 0,                                                 0, -1,  0,  0,  0,  0,    0, 0],
            [0, 0, 0, 0,                         0, 0,                                                 0,  0, -1,  0,  0,  0,    0, 0],
            [0, 0, 0, 0, lomega*x4*x5*(-x10 - x11), 0,                                               x14,  0,  0,  0,  0,  0,   x9, 0],
            [0, 0, 0, 0,                         0, 0,                                                 0,  0,  0,  0,  0, -1,    0, 0],
            [0, 0, 0, 0,                       x14, 0, 2*c1*lomega*u*x12*x15 - lvx*ut1*x16 - lvy*ut2*x16,  0,  0, x2, x3,  0, -x13, 0]
        ], float)
    @staticmethod
    def Hamiltonian(fullstate, control, c1, c2, c3, g):
        x,y,vx,vy,theta,omega,m,lx,ly,lvx,lvy,ltheta,lomega,lm = fullstate
        u, ut1, ut2 = control
        x0 = 1/c2
        x1 = c1*u*x0
        x2 = 1/m
        x3 = c1*u*ut1*x2
        return  a*x1 + c1**2*u**2*x0*(-a + 1) - c1*lomega*u*x2*(ut1*sin(theta) + ut1*cos(theta))/c3 - lm*x1 + ltheta*omega + lvx*x3 + lvy*(-g + x3) + lx*vx + ly*vy
    @staticmethod
    def Pontryagin(fullstate, c1, c2, c3, g, a):
        x,y,vx,vy,theta,omega,m,lx,ly,lvx,lvy,ltheta,lomega,lm = fullstate
        x0  = lomega/c3
        x1  = x0*cos(theta)
        x2  = x0*sin(theta)
        x3  = 1/sqrt((lvx - x1)**2 + (lvy + x2)**2)
        ut1 = x3*(-lvx + x1)
        ut2 = x3*(-lvy - x2)
        ang = arctan2(ut1, ut2)
        ang = ang-theta
        lim = 10/360*2*pi
        th  = min(max(lim, ang), lim)
        ut1 = sin(th)
        ut2 = cos(th)
        # TODO: Finish this! ..zz.. zzzz....zzz.

''' ------
Propogator
------ '''
class Propagate(object):
    def __init__(self, model):
        self.model = model
    def Ballistic(self, si=None, tf=None, nnodes=None):
        if si is None: si = self.model.si
        if tf is None: tf = self.model.tub
        if nnodes is None: nnodes = 20
        return odeint(
            self.EOM,
            si,
            linspace(0, tf, nnodes),
            Dfun = self.EOM_Jac,
            rtol = 1e-12,
            atol = 1e-12,
            args = (zeros(self.model.cdim),) # No control
        )
    def Indirect(self, fsi, tf, nnodes):
        nsegs = nnodes - 1
        t    = linspace(0, tf, nnodes)
        fs   = array(fsi, ndmin=2)
        c    = array(self.model.Pontryagin(fsi), ndmin=2)
        # Must integrate incrimentally to track control
        for k in range(nsegs):
            fskp1 = odeint(
                self.EOM_Indirect,
                fs[k],
                [t[k], t[k+1]],
                Dfun = self.EOM_Indirect_Jac,
                rtol = 1e-13,
                atol = 1e-13
            )
            fskp1 = fskp1[1]
            fs = vstack((fs, fskp1))
            c  = vstack((c, self.model.Pontryagin(fskp1)))
        return t, fs, c
    def EOM(self, state, t, control):
        return self.model.EOM_State(state, control)
    def EOM_Jac(self, state, t, control):
        return self.model.EOM_State_Jac(state, control)
    def EOM_Indirect(self, fullstate, t):
        control = self.model.Pontryagin(fullstate)
        return self.model.EOM_Fullstate(fullstate, control)
    def EOM_Indirect_Jac(self, fullstate, t):
        control = self.model.Pontryagin(fullstate)
        return self.model.EOM_Fullstate_Jac(fullstate, control)

if __name__ == "__main__":
