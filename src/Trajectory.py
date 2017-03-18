'''
Astro.IQ - Optimisation
Christopher Iliffe Sprague
christopher.iliffe.sprague@gmail.com
https://cisprague.github.io/Astro.IQ
'''

''' --------
Dependencies
-------- '''
from numpy import *
from scipy.integrate import odeint

''' -----------
Dynamical Model
----------- '''
class Dynamical_Model(object):
    def __init__(self, si, st, slb, sub, clb, cub, tlb, tub):
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
    def Safe(self, state):
        for cond in state <= self.sub:
            if cond: pass
            else: return False
        for cond in state >= self.lub:
            if cond: pass
            else: return False
        return True
    # For Indirect Methods
    def EOM_Fullstate(self, fullstate, control):
        return None
    def EOM_Fullstate_Jac(self, fullstate, control):
        return None
    def Hamiltonian(self, fullstate):
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
        T   = 44000):
        # Problem parameters
        self.Isp  = float(Isp)   # Specific impulse [s]
        self.g    = float(g)     # Environment's gravity [m/s^2]
        self.T    = float(T)     # Maximum thrust [N]
        self.g0   = float(9.802) # Earth's sea-level gravity [m/s^2]
        self.a    = float(0)
        # For optimisation
        Dynamical_Model.__init__(
            self,
            si,
            st,
            [-1000, 0, -500, -500, 0],
            [1000, 2000, 500, 500, 10000],
            [-0.001, -1, -0.001],
            [1, 1, 1],
            1,
            200
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
    def EOM_Fullstate(self, fullstate, control):
        x, y, vx, vy, m, lx, ly, lvx, lvy, lm = fullstate
        u, st, ct     = control
        T, Isp, g0, g = self.T, self.Isp, self.g0, self.g
        x0            = T*u/m
        x1            = T*u/m**2
        return array([
            [                   vx],
            [                   vy],
            [                st*x0],
            [            ct*x0 - g],
            [        -T*u/(Isp*g0)],
            [                    0],
            [                    0],
            [                  -lx],
            [                  -ly],
            [ct*lvy*x1 + lvx*st*x1]
        ], float)
    def EOM_Fullstate_Jac(self, fullstate, control):
        x, y, vx, vy, m, lx, ly, lvx, lvy, lm = fullstate
        u, st, ct = control
        T         = self.T
        x0        = T*u/m**2
        x1        = st*x0
        x2        = ct*x0
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
            [0, 0, 0, 0, -ct*lvy*x3 - lvx*st*x3,  0,  0, x1, x2, 0]
        ], float)
    def Hamiltonian(self, fullstate):
        x, y, vx, vy, m, lx, ly, lvx, lvy, lm = fullstate
        T, Isp, g0, g = self.T, self.Isp, self.g0, self.g
        x0 = T*u/m
        x1 = 1/(Isp*g0)
        H  = -T*lvm*u*x1 + lvx*st*x0 + lvy*(ct*x0 - g) + lx*vx + ly*vy
        H += x1*(T**2*u**2*(-a + 1) + a*(T*u)) # The Lagrangian
        return H
    def Pontryagin(self, fullstate):
        x, y, vx, vy, m, lx, ly, lvx, lvy, lm = fullstate
        lv = sqrt(abs(lvx)**2 + abs(lvy)**2)
        st = -lvx/lv
        ct = -lvy/lv
        u  = -self.Isp*self.g0*lv/m + lm - self.a
        u /= 2*self.T*(1 - self.a)
        u  = max(u, 0)
        u  = min(u, 1)
        return u, st, ct


class Rocket_Lander(Dynamical_Model):
    def __init__(self):
        return None

''' -
Space
- '''
class CRTBP(Dynamical_Model):
    def __init__(
        self,
        mu  = 1.21506683e-2,
        T   = 10.,
        Isp = 3000.,
        eps = 1.,
        si  = array([-0.019488511458668, -0.016033479812051, 0,
                    8.918881923678198, -4.081793688818725, 0,
                    1000], float),
        st  = array([0.823385182067467, 0, -0.022277556273235,
                    0, 0.134184170262437, 0,
                    1000], float)
        ):
        self.mu  = float(mu)
        self.T   = float(T)
        self.Isp = float(Isp)
        self.g0  = 9.80665
        self.eps = eps
        self.si  = si
        self.st  = st
        super(CRTBP, self).__init__(8, 0, 1, 8, 0, 1e-3)
        self.set_bounds([5.] + [-100.]*7, [300.] + [100.]*7)
    def EOM(self, t, fullstate, control=True):
        x, y, z, vx, vy, vz, m, lx, ly, lz, lvx, lvy, lvz, lm = fullstate
        T, mu, Isp, g0, eps = self.T, self.mu, self.Isp, self.g0, self.eps
        if control is True:
            u, ax, ay, az, S = self.Pontryagin(fullstate)
        elif control is False:
            u, ax, az, ay = 0, 0, 0, 0
        x0  = T*u/m
        x1  = mu - 1
        x2  = x + x1
        x3  = y**2
        x4  = z**2
        x5  = x3 + x4
        x6  = x2**2 + x5
        x7  = mu/x6**(3/2.)
        x8  = mu + x
        x9  = -mu + 1
        x10 = x5 + x8**2
        x11 = x10**(-3/2.)
        x12 = x11*x9
        x13 = x1*x11
        x14 = -3*mu - 3*x
        x15 = x10**(-5/2.)
        x16 = x6**(-5/2.)
        x17 = mu*x16*(x14 + 3)
        x18 = x14*x15*x9
        x19 = -x7
        x20 = -x12 + x19 + 1
        x21 = y*z
        x22 = 3*mu*x16
        x23 = x21*x22
        x24 = 3*x1*x15
        x25 = 3*mu*x16*x2
        x26 = 3*x15*x8*x9
        x27 = 3*x15*x9
        x28 = T*u/m**2
        return array([
            vx,
            vy,
            vz,
            ax*x0+2*vy+x-x12*x8-x2*x7,
            ay*x0-2*vx-x12*y-x7*y+y,
            az*x0+x13*z-x7*z,
            -T*u/(Isp*g0),
            -lvx*(-x17*x2-x18*x8+x20)-lvy*(-x17*y-x18*y)-lvz*(x1*x14*x15*z-x17*z),
            -lvx*(x25*y+x26*y)-lvy*(x20+x22*x3+x27*x3)-lvz*(-x21*x24+x23),
            -lvx*(x25*z+x26*z)-lvy*(x21*x27+x23)-lvz*(x13+x19+x22*x4-x24*x4),
            2*lvy-lx,
            -2*lvx-ly,
            -lz,
            ax*lvx*x28+ay*lvy*x28+az*lvz*x28
        ], float)
    def EOM_Jac(self, t, fullstate, control=True):
        mu, T, Isp = self.mu, self.T, self.Isp
        if control is True:
            u, ax, ay, az, S = self.Pontryagin(fullstate)
        elif control is False:
            u, ax, ay, az = 0, 0, 0, 0
        x0  = mu - 1
        x1  = x + x0
        x2  = y**2
        x3  = z**2
        x4  = x2 + x3
        x5  = x1**2 + x4
        x6  = mu/x5**(3/2.)
        x7  = -x6
        x8  = -mu + 1
        x9  = mu + x
        x10 = x4 + x9**2
        x11 = x10**(-3/2.)
        x12 = x11*x8
        x13 = -x12 + x7 + 1
        x14 = 3*mu
        x15 = -3*x - x14
        x16 = x15 + 3
        x17 = x5**(-5/2.)
        x18 = mu*x16*x17
        x19 = x1*x18
        x20 = x10**(-5/2.)
        x21 = x15*x20
        x22 = x21*x8
        x23 = x22*x9
        x24 = x17*y
        x25 = x14*x24
        x26 = x1*x25
        x27 = x8*y
        x28 = 3*x20*x27
        x29 = x28*x9
        x30 = x17*z
        x31 = x14*x30
        x32 = x1*x31
        x33 = 3*x20*x8
        x34 = x33*z
        x35 = x34*x9
        x36 = T*u/m**2
        x37 = ax*x36
        x38 = mu*x16
        x39 = x24*x38
        x40 = x21*x27
        x41 = 3*mu*x17
        x42 = x2*x41
        x43 = x2*x33
        x44 = x25*z
        x45 = x28*z
        x46 = ay*x36
        x47 = x0*z
        x48 = x21*x47
        x49 = x30*x38
        x50 = 3*x0*x20
        x51 = x50*y
        x52 = x51*z
        x53 = x0*x11
        x54 = x3*x41
        x55 = x3*x50
        x56 = az*x36
        x57 = -x50*z
        x58 = -5*mu - 5*x
        x59 = x10**(-7/2.)
        x60 = x15*x58*x59
        x61 = x58 + 5
        x62 = x5**(-7/2.)
        x63 = mu*x16*x61*x62
        x64 = x25 + x28
        x65 = x1*x41 + x33*x9
        x66 = 5*mu*x16
        x67 = x62*y*z
        x68 = x66*x67
        x69 = -15*mu - 15*x
        x70 = x0*x59*y*z
        x71 = x1*x62*y
        x72 = -x18
        x73 = -x22 + x72
        x74 = x2*x62
        x75 = x31 + x34
        x76 = x1*x62*z
        x77 = x0*x21 + x72
        x78 = x3*x62
        x79 = x12 + x6 - 1
        x80 = 3*mu*x61
        x81 = x67*x80
        x82 = 15*mu*x62*z
        x83 = -x2*x82 + x31
        x84 = 9*mu
        x85 = y**3
        x86 = 15*mu*x62
        x87 = 9*x20
        x88 = 15*x59*x8
        x89 = 15*mu*x1*x62
        x90 = 15*x2*x59*x8
        x91 = 15*x59*x8*y
        x92 = x3*y
        x93 = x25 - x86*x92
        x94 = 15*x0*x59
        x95 = -lvx*(-x1*x82*y - x9*x91*z) - lvy*(x34 + x83 - x90*z) - lvz*(-x51 + x92*x94 + x93)
        x96 = -x44
        x97 = z**3
        x98 = 2*T*u/m**3
        return array([
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [x13-x19-x23, x26+x29, x32+x35, 0, 2, 0, -x37, 0, 0, 0, 0, 0, 0, 0],
            [-x39-x40, x13+x42+x43, x44+x45, -2, 0, 0, -x46, 0, 0, 0, 0, 0, 0, 0],
            [x48-x49, x44-x52, x53+x54-x55+x7, 0, 0, 0, -x56, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-lvx*(-x1*x63-2*x18-2*x22-x60*x8*x9+x65)-lvy*(-x27*x60-x63*y+x64)-lvz*(x31+x47*x60+x57-x63*z), -lvx*(x59*x69*x8*x9*y+x64+x66*x71)-lvy*(x2*x59*x69*x8+x66*x74+x73)-lvz*(x68-x69*x70), -lvx*(x59*x69*x8*x9*z+x66*x76+x75)-lvy*(x59*x69*x8*y*z+x68)-lvz*(-x0*x3*x59*x69+x66*x78+x77), 0, 0, 0, 0, 0, 0, 0, x19+x23+x79, x39+x40, -x48+x49, 0],
            [-lvx*(x59*x69*x8*x9*y+x64+x71*x80)-lvy*(x2*x59*x69*x8+x73+x74*x80)-lvz*(-x69*x70+x81), -lvx*(-x2*x89+x65-x9*x90)-lvy*(x24*x84+x27*x87-x85*x86-x85*x88)-lvz*(15*x2*x47*x59+x57+x83), x95, 0, 0, 0, 0, 0, 0, 0, -x26-x29, -x42-x43+x79, x52+x96, 0],
            [-lvx*(x59*x69*x8*x9*z+x75+x76*x80)-lvy*(x59*x69*x8*y*z+x81)-lvz*(-x0*x3*x59*x69+x77+x78*x80), x95, -lvx*(-x3*x88*x9-x3*x89+x65)-lvy*(x28-x3*x91+x93)-lvz*(x30*x84-x47*x87-x86*x97+x94*x97), 0, 0, 0, 0, 0, 0, 0, -x32-x35, -x45+x96, -x53-x54+x55+x6, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -ax*lvx*x98-ay*lvy*x98-az*lvz*x98, 0, 0, 0, x37, x46, x56, 0]
        ], float)
    def Shoot(self, decision, control=True):
        tf, lx, ly, lz, lvx, lvy, lvz, lm = decision
        x, y, z, vx, vy, vz, m = self.si
        fsi = array([x, y, z, vx, vy, vz, m, lx ,ly, lz, lvx, lvy, lvz, lm])
        solver = ode(self.EOM, self.EOM_Jac)
        solver.set_integrator('dop853', nsteps=1, atol=1e-8, rtol=1e-8)
        solver.set_initial_value(fsi, 0)
        solver.set_f_params(control)
        solver.set_jac_params(control)
        t, traj, cont = [], [], []
        while solver.t < tf:
            solver.integrate(tf, step=True)
            t.append(solver.t)
            traj.append(solver.y)
            cont.append(self.Pontryagin(solver.y))
        return array(t), array(traj), array(cont)
    def Pontryagin(self, fullstate):
        x, y, z, vx, vy, vz, m, lx, ly, lz, lvx, lvy, lvz, lm = fullstate
        Isp = self.Isp
        g0  = self.g0
        eps = self.eps
        # Switching function
        S = -Isp*g0*sqrt(abs(lvx)**2 + abs(lvy)**2 + abs(lvz)**2)/m - lm + 1
        if S > eps:
            u = 0.
        elif -eps <= S and S <= eps:
            u = (eps - S)/(2*eps)
        elif S < -eps:
            u = 1.
        # Optimal thrust direction
        ax = -lvx/sqrt(abs(lvx)**2 + abs(lvy)**2 + abs(lvz)**2)
        ay = -lvy/sqrt(abs(lvx)**2 + abs(lvy)**2 + abs(lvz)**2)
        az = -lvz/sqrt(abs(lvx)**2 + abs(lvy)**2 + abs(lvz)**2)
        return array([u, ax, ay, az, S], float)
    def Hamiltonian(self, fullstate, control=True):
        x, y, z, vx, vy, vz, m, lx, ly, lz, lvx, lvy, lvz, lm = fullstate
        T   = self.T
        mu  = self.mu
        Isp = self.Isp
        g0  = self.g0
        eps = self.eps
        if control is True:
            u, ax, ay, az, S = self.Pontryagin(fullstate)
        elif control is False:
            u, ax, ay, az = 0, 0, 0, 0
        x0 = T/(Isp*g0)
        x1 = T*u/m
        x2 = mu - 1
        x3 = x + x2
        x4 = y**2 + z**2
        x5 = mu/(x3**2 + x4)**(3/2)
        x6 = mu + x
        x7 = (x4 + x6**2)**(-3/2)
        x8 = x7*(-mu + 1)
        return -lm*u*x0+lvx*(ax*x1+2*vy+x-x3*x5-x6*x8)+lvy*(ay*x1-2*vx-x5*y-x8*y+y)+lvz*(az*x1+x2*x7*z-x5*z)+lx*vx+ly*vy+lz*vz+x0*(-eps*u*(-u+1)+u)

''' ------
Propogator
------ '''
class Propagate(object):
    def __init__(self, model):
        self.model = model
    def Ballistic_Beta(self, si=None, tf=None, adaptive=True):
        if si is None: si = self.model.si
        if tf is None: tf = self.model.tub
        solver = ode(self.EOM_Ballistic, self.EOM_Ballistic_Jac)
        solver.set_initial_value(si, 0)
        solver.set_f_params(zeros(self.model.cdim))
        if adaptive:
            solver.set_integrator('dop853', nsteps=1, atol=1e-14, rtol=1e-14, beta=0.1) # RK 8(5,3)
        else:
            pass
        t, state = [], []
        while self.model.Safe(solver.y) and solver.t < tf:
            solver.integrate(tf, step=True)
            t.append(solver.t)
            state.append(solver.y)
        return array(t), array(state)
    def Ballistic(self, si=None, tf=None, nnodes=None):
        if si is None: si = self.model.si
        if tf is None: tf = self.model.tub
        return odeint(
            self.EOM,
            si,
            linspace(0,tf, nnodes),
            Dfun = self.EOM_Jac,
            rtol = 1e-12,
            atol = 1e-12,
            args = (zeros(self.model.cdim),) # No control
        )
    def EOM(self, state, t, control):
        return self.model.EOM_State(state, control)
    def EOM_Jac(self, state, t, control):
        return self.model.EOM_State_Jac(state, control)

if __name__ == "__main__":
    mod = Point_Lander()
    fs = array(list(mod.sub) + [1,1,1,1,1])
    print mod.Pontryagin(fs)
