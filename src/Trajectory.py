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
import matplotlib.pyplot as plt
set_printoptions(suppress=True)

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
        # Lower and upper initial state bounds
        self.silb, self.siub = array(silb, float), array(siub, float)
        # Numerical integration
        self.Propagate       = Propagate(self)
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
    def EOM_State(self, state, control):
        return None
    def EOM_State_Jac(self, state, control):
        return None
    def EOM_Fullstate(self, fullstate, control):
        return None
    def EOM_Fullstate_Jac(self, fullstate, control):
        return None
    def Hamiltonian(self, fullstate, control):
        return None
    def Pontryagin(self, fullstate):
        return None

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
        self.g0   = float(9.81)  # Earth's sea-level gravity [m/s^2]
        self.a    = float(a)     # Homotopy parametre
        # For optimisation
        Dynamical_Model.__init__(
            self,
            si,
            st,
            [-10000, 0, -500, -500, 10],
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
        u, ux, uy      = control
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
class Point_Lander_Drag(Dynamical_Model):
    # Specs taken from SpaceX Dragon 2
    def __init__(
        self,
        si  = [0, 5000, 150, -10, 8165],
        st  = [0, 0, 0, 0, 5000],       # Final mass doesn't matter
        T   = 68.17e3,                  # SuperDraco [N]
        Isp = 243.,                     # SuperDraco [s]
        CD  = 1.4,                      # Dragon2
        rho = 0.020,                    # Mars [kg/m^3]
        A   = 10.52,                    # Phoenix [m^2]
        g   = 3.711):                   # Mars [m/s^2]
        # Problem parametres
        self.c1 = float(T)              # Max Thrust
        self.c2 = float(Isp*9.81)       # Effective Velocity
        self.c3 = float(0.5*rho*CD*A)   # Aerodynamics
        self.g  = float(g)              # Gravity
        # Initialise the model
        Dynamical_Model.__init__(
            self,
            si,
            st,
            [-20000, 0, -200, -200, 10],
            [20000, 5000, 200, 0, 10000],
            [0, 0], # NOTE: theta is measured from the horrizontal
            [1, pi], # we restrict the thrust to only point upward
            1,
            1000,
            [-500, 1000, -300, -300, 6585],
            [500, 5000, 300, 0, 8165])
    def EOM_State(self, state, control):
        x, y, vx, vy, m = state
        u, theta        = control
        c1, c2, c3, g   = self.c1, self.c2, self.c3, self.g
        x0              = 1/m
        x1              = c1*u*x0
        x2              = c3*x0*sqrt(abs(vx)**2 + abs(vy)**2)
        return array([
            vx,
            vy,
            -vx*x2 + x1*cos(theta),
            -g - vy*x2 + x1*sin(theta),
            -c1*u/c2
        ], float)
    def Plot(self, state, control):
        # Plot methods are specific to model
        x, y, vx, vy, m = hsplit(state, self.sdim)
        u, theta        = hsplit(control, self.cdim)
        plt.figure(1)
        # Trajectory
        plt.subplot(131)
        plt.plot(x, y, 'k.-')
        plt.xlabel('Cross-Range [m]')
        plt.ylabel('Altitude [m]')
        # Velocities
        plt.subplot(232)
        plt.plot(vx, 'k.-')
        plt.plot(vy, 'k.--')
        plt.legend(['$v_x$', '$v_y$'], loc='best')
        plt.xlabel('Node Index')
        plt.ylabel('Velocity [m/s]')
        # Mass
        plt.subplot(233)
        plt.plot(m, 'k.-')
        plt.xlabel('Node Index')
        plt.ylabel('Mass [kg]')
        # Throttle
        plt.subplot(235)
        plt.plot(u, 'k.-')
        plt.xlabel('Node Index')
        plt.ylabel('Throttle')
        # Thrust angle
        plt.subplot(236)
        plt.plot(theta, 'k.-')
        plt.xlabel('Node Index')
        plt.ylabel('Thrust Angle [rad]')
        plt.tight_layout()
        plt.show()



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
    model  = Point_Lander_Drag()
    print model.c1
