from numpy import array, sqrt, hstack, linspace, meshgrid, arange, outer, ones, cos, sin, pi, size, concatenate
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
    '''
        Indirect trajectory optimization of circular restricted
        three body problem (CRTBP). The trajectory optimisation
        problem is to select the intial costates to lead the
        trajectory optimally through Pontryagin's Minimum Principle.
    '''
    # TODO: 1) Nicer plots, 2) Use hybdrid particle swarm optimisation
    def __init__(self):
        self.mu = 0.01215 # Mass ratio: mu = m2/(m1+m2)
        self.T  = 400 # Max thrust
        self.Isp = 300
        self.g0  = 9.8124
        self.eps = 1
        self.s0 = array([-0.3, 0, 0, 0, 0, 0, 1000], float)
        self.L1 = (0.836915, 0)
        self.L2 = (1.155682, 0)
        self.L3 = (-1.005063, 0)
        self.L4 = (0.4878494, 0.866025)
        self.L5 = (0.4878494, -0.866025)
        self.S  = []
        self.U  = []
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
        solver = ode(self.EOM).set_integrator('dop853',nsteps=1, atol=1e-14, rtol=1e-14)
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
            f, ax = plt.subplots(2,2)
            ax[0, 0].plot(traj[:,0], traj[:,1], 'k.-')
            ax[0, 0].plot([-self.mu], [0], 'ko')
            ax[0, 0].plot([1-self.mu], [0], 'ko')
            ax[0, 0].plot([0.836915], [0], 'kx')
            ax[0, 0].plot([1.155682], [0], 'kx')
            ax[0, 0].plot([-1.005063], [0], 'kx')
            ax[0, 0].plot([0.4878494], [0.866025], 'kx')
            ax[0, 0].plot([0.4878494], [-0.866025], 'kx')
            ax[0, 1].plot(self.S)
            ax[1,0].plot(self.U)
            ax[1,1].plot(traj[:,6:13])
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
    def EOM_State(self, state, control):
        x, y, z, vx, vy, vz, m = state
        u, ax, ay, az = control
        mu, T, g0, Isp = self.mu, self.T, self.g0, self.Isp
        return array([
            vx,
            vy,
            vz,
            (ax*T*u)/m+2*vy+x-(mu*(-1+mu+x))/((-1+mu+x)**2+y**2+z**2)**(3/2.)-((1-mu)*(mu+x))/((mu+x)**2+y**2+z**2)**(3/2.),
            (ay*T*u)/m-2*vx+y-(mu*y)/((-1+mu+x)**2+y**2+z**2)**(3/2.)-((1-mu)*y)/((mu+x)**2+y**2+z**2)**(3/2.),
            (az*T*u)/m-(mu*z)/((-1+mu+x)**2+y**2+z**2)**(3/2.)+((-1+mu)*z)/((mu+x)**2+y**2+z**2)**(3/2.),
            -((T*u)/(g0*Isp))
        ], float)
    def EOM_Costate(self, fullstate, control):
        x, y, z, vx, vy, vz, m, lx, ly, lz, lvx, lvy, lvz, lvm = fullstate
        u, ax, ay, az = control
        mu, T = self.mu, self.T
        return array([
            -(lvy*((3*mu*(-1+mu+x)*y)/((-1+mu+x)**2+y**2+z**2)**(5/2.)+(3*(1-mu)*(mu+x)*y)/((mu+x)**2+y**2+z**2)**(5/2.)))-lvz*((3*mu*(-1+mu+x)*z)/((-1+mu+x)**2+y**2+z**2)**(5/2.)-(3*(-1+mu)*(mu+x)*z)/((mu+x)**2+y**2+z**2)**(5/2.))-lvx*(1+(3*mu*(-1+mu+x)**2)/((-1+mu+x)**2+y**2+z**2)**(5/2.)-mu/((-1+mu+x)**2+y**2+z**2)**(3/2.)+(3*(1-mu)*(mu+x)**2)/((mu+x)**2+y**2+z**2)**(5/2.)-(1-mu)/((mu+x)**2+y**2+z**2)**(3/2.)),
            -(lvx*((3*mu*(-1+mu+x)*y)/((-1+mu+x)**2+y**2+z**2)**(5/2.)+(3*(1-mu)*(mu+x)*y)/((mu+x)**2+y**2+z**2)**(5/2.)))-lvz*((3*mu*y*z)/((-1+mu+x)**2+y**2+z**2)**(5/2.)-(3*(-1+mu)*y*z)/((mu+x)**2+y**2+z**2)**(5/2.))-lvy*(1+(3*mu*y**2)/((-1+mu+x)**2+y**2+z**2)**(5/2.)-mu/((-1+mu+x)**2+y**2+z**2)**(3/2.)+(3*(1-mu)*y**2)/((mu+x)**2+y**2+z**2)**(5/2.)-(1-mu)/((mu+x)**2+y**2+z**2)**(3/2.)),
            -(lvx*((3*mu*(-1+mu+x)*z)/((-1+mu+x)**2+y**2+z**2)**(5/2.)+(3*(1-mu)*(mu+x)*z)/((mu+x)**2+y**2+z**2)**(5/2.)))-lvy*((3*mu*y*z)/((-1+mu+x)**2+y**2+z**2)**(5/2.)+(3*(1-mu)*y*z)/((mu+x)**2+y**2+z**2)**(5/2.))-lvz*((3*mu*z**2)/((-1+mu+x)**2+y**2+z**2)**(5/2.)-mu/((-1+mu+x)**2+y**2+z**2)**(3/2.)-(3*(-1+mu)*z**2)/((mu+x)**2+y**2+z**2)**(5/2.)+(-1+mu)/((mu+x)**2+y**2+z**2)**(3/2.)),
            2*lvy-lx,
            -2*lvx-ly,
            -lz,
            (ax*lvx*T*u)/m**2+(ay*lvy*T*u)/m**2+(az*lvz*T*u)/m**2
        ], float)
    def Pontryagin(self, fullstate):
        x, y, z, vx, vy, vz, m, lx, ly, lz, lvx, lvy, lvz, lvm = fullstate
        g0 = self.g0
        Isp = self.Isp
        eps = self.eps
        ax = -(lvx/sqrt(abs(lvx)**2+abs(lvy)**2+abs(lvz)**2))
        ay = -(lvy/sqrt(abs(lvx)**2+abs(lvy)**2+abs(lvz)**2))
        az = -(lvz/sqrt(abs(lvx)**2+abs(lvy)**2+abs(lvz)**2))
        S  = 1-lm-(g0*Isp*sqrt(abs(lvx)**2+abs(lvy)**2+abs(lvz)**2))/m
        self.S.append(S)
        if S > eps:
            u = 0
        elif -eps <= S and S <= eps:
            u = (eps - S)/2*eps
        elif S < -eps:
            u = 1
        self.U.append(u)
        print u
        return array([u, ax, ay, az], float)
    def EOM(self, t, fullstate):
        state   = fullstate[0:7]
        control = self.Pontryagin(fullstate)
        ds      = self.EOM_State(state, control)
        dl      = self.EOM_Costate(fullstate, control)
        return hstack((ds, dl))

if __name__ == "__main__":
    '''
    from sympy.parsing import mathematica
    print mathematica.parse('1 - lm - (g0*Isp*Sqrt[Abs[lvx]^2 + Abs[lvy]^2 + Abs[lvz]^2])/m')
    '''

    p = CRTBP()
    s = p.s0
    c = array([0.5, 1/3., 2/3., 2/3.], float)
    l = array([15, 32, -0.945, -0.101, 0.04479, -0.00015, 0.1332], float)
    lx,ly,lz,lvx,lvy,lvz,lm = l
    x,y,z,vx,vy,vz,m = s
    u, ax, ay, az = c
    fs = array([x,y,z,vx,vy,vz,m,lx,ly,lz,lvx,lvy,lvz,lm])
    traj = p.Shoot(fs, 10)
    p.PlotTraj(traj)
    plt.show()
