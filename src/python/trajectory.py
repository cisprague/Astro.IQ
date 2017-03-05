from numpy import array, sqrt, hstack, linspace, meshgrid, arange, outer, ones, cos, sin, pi, size
from numpy.linalg import norm
from scipy.integrate import ode, odeint
import matplotlib.pyplot as plt
from PyGMO.problem import base, normalized
from PyGMO import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator
import matplotlib.animation as animation
import matplotlib.cm as cm

class PointLander2D(base):
    def __init__(
        self,
        s0 = array([0,1000,-5,0,10000],dtype=float),
        st = array([0,0,0,0,0],dtype=float),
        c1 = 44000.,
        c2 = 311.*9.81,
        g  = 1.6229,
        a = 0.0
    ):
        self.s0    = s0
        self.st    = st
        self.c1    = c1
        self.c2    = c2
        self.g     = g
        self.a     = a
        # PyGMO NLP: Z = [tf,lx,ly,lvx,lvy,lm]
        super(PointLander2D, self).__init__(6,0,1,6,0,1e-5)
        self.set_bounds([1]+[-1e6]*5, [10000]+[1e6]*5)
        return None
    def EOM_State(self, state, control):
        x,y,vx,vy,m = state
        u,st,ct     = control
        dx          = vx
        dy          = vy
        dvx         = self.c1*u*st/m
        dvy         = self.c1*u*ct/m - self.g
        dm          = -self.c1*u/self.c2
        return array([dx,dy,dvx,dvy,dm],dtype=float)
    def EOM_Costate(self,fullstate,control):
        x,y,vx,vy,m,lx,ly,lvx,lvy,lm = fullstate
        u,st,ct = control
        dlx  = 0.
        dly  = 0.
        dlvx = -lx
        dlvy = -ly
        dlm  = self.c1*u*(lvx*st+lvy*ct)/m**2
        return array([dlx,dly,dlvx,dlvy,dlm],dtype=float)
    def Pontryagin(self,fullstate):
        x,y,vx,vy,m,lx,ly,lvx,lvy,lm = fullstate
        lv = norm([lvx,lvy])
        st = -lvx/lv
        ct = -lvy/lv
        if self.a == 1:
            S = 1. - lm - lv*self.c2/m
            if S >= 0:
                u = 0
            elif S < 0:
                u = 1
        else:
            u = min(max((lm - self.a - lv*self.c2/m)/(2*self.c1*(1-self.a)),0),1)
        return array([u,st,ct],dtype=float)
    def EOM(self,fullstate,t):
        state = fullstate[:5]
        control = self.Pontryagin(fullstate)
        ds = self.EOM_State(state, control)
        dl = self.EOM_Costate(fullstate, control)
        return hstack((ds, dl))
    def Shoot(self,decision):
        tf = decision[0]
        l0 = decision[1:]
        fs0 = hstack((self.s0,l0))
        t = linspace(0,tf,100)
        sol= odeint(self.EOM, fs0, t)
        return t, sol
    def Hamiltonian(self,fullstate):
        s = fullstate[:5]
        l = fullstate[5:]
        c = self.Pontryagin(fullstate)
        ds = self.EOM_State(s,c)
        H = 0.
        for li, dsi in zip(l,s):
            H += li*dsi
        u,st,ct = c
        H += ((1-self.a)*self.c1**2*u**2 + self.a*self.c1*u)/self.c2
        return H
    # PyGMO Stuff (Thanks ESA!)
    def _objfun_impl(self, decision):
        return (1.,)
    def _compute_constraints_impl(self, decision):
        t, fsf = self.Shoot(decision)
        fsf = fsf[-1]
        xf,yf,vxf,vyf,mf,lxf,lyf,lvxf,lvyf,lmf = fsf
        xt,yt,vxt,vyt,mt = self.st
        H = self.Hamiltonian(fsf)
        return [xf-xt,yf-yt,vxf-vxt,vyf-vyt,lmf,H]
    def Plot(self, traj, t):
        f, ax = plt.subplots(2,2)
        ax[0,0].plot(traj[:,0],traj[:,1])
        ax[0,1].plot(t,traj[:,2],t,traj[:,3])
        ax[1,0].plot(t,traj[:,4])
        plt.show()

class CRTBP(object):
    def __init__(self):
        self.mu = 0.01215
        self.s0 = array([-0.6, 0.7, 0, 0, 0, 0, 0], dtype = float)

    def EOM_State(self, t, state):
        x, y, z, vx, vy, vz, m = state
        mu  = self.mu
        R1  = self.R1(x, y, z)
        R2  = self.R2(x, y, z)
        dx  = vx
        dy  = vy
        dz  = vz
        dvx = 2*vy + x - (1-mu)*(x+mu)/R1**3 - mu*(x-1+mu)/R2**3
        dvy = -2*vx + y - (1-mu)*y/R1**3 - mu*y/R2**3
        dvz = -(1-mu)*z/R1**3 - mu*z/R2**3
        dvm = 0
        return array([dx, dy, dz, dvx, dvy, dvz, dvm], dtype = float)

    def JacobiConstant(self, x, y, z, vx, vy, vz):
        V = norm([vx, vy, vz])
        Omega = self.PsuedoPotential(x, y, z)
        return 2*Omega - V**2

    def PsuedoPotential(self, x, y, z):
        R1 = self.R1(x, y, z)
        R2 = self.R2(x, y, z)
        return 0.5*(x**2+y**2) + (1-self.mu)/R1 + self.mu/R2

    def R1(self, x, y, z):
        return ((x+self.mu)**2 + y**2 + z**2)**0.5

    def R2(self, x, y, z):
        return ((x-1+self.mu)**2 + y**2 + z**2)**0.5

    def Shoot(self, state0, tf):
        solver = ode(self.EOM_State).set_integrator('dop853',nsteps=1, atol=1e-14, rtol=1e-14)
        solver.set_initial_value(state0, 0)
        sol = []
        while solver.t < tf:
            solver.integrate(tf, step=True)
            sol.append(solver.y)
        return array(sol)

    def PlotTraj(self, traj, dim = '2d'):
        if dim is '3d':
            fig = plt.figure()
            ax  = fig.add_subplot(111, projection='3d')
            ax.plot(traj[:,0], traj[:,1], traj[:,2], 'k-')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.grid(False)
            ax.xaxis.pane.set_edgecolor('black')
            ax.yaxis.pane.set_edgecolor('black')
            ax.zaxis.pane.set_edgecolor('black')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis._axinfo['tick']['inward_factor'] = 0
            ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
            ax.yaxis._axinfo['tick']['inward_factor'] = 0
            ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
            ax.zaxis._axinfo['tick']['inward_factor'] = 0
            ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
            ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
            plt.show()
        elif dim is '2d':
            plt.plot(traj[:,0], traj[:,1], 'k.-')
            plt.plot([-self.mu], [0], 'ko')
            plt.plot([1-self.mu], [0], 'ko')
            plt.xlabel('X')
            plt.ylabel('Y')
        return None

    def PlotJacoabiConstant(self):
        x    = arange(-1.5, 1.5, 0.01)
        y    = arange(-1.5, 1.5, 0.01)
        x, y = meshgrid(x, y)
        z  = self.JacobiConstant(x, y, 0, 0, 0, 0)
        CS = plt.contourf(x, y, z, levels = arange(2, 4, 0.1), cmap = cm.gray)
        CL = plt.contour(x, y, z, levels = arange(2, 4, 0.1), colors = 'k')
        plt.colorbar(CS)
        return None

if __name__ == "__main__":
    p = CRTBP()
    traj = p.Shoot(p.s0, 100)
    p.PlotTraj(traj,'2d')
    p.PlotJacoabiConstant()
    plt.show()
