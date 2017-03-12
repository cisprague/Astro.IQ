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

if __name__ == "__main__":
    from PyGMO import *
    prob = PointLander2D()
    print population(prob, 10)
