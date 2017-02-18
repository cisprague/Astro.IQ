from PyGMO.problem._base import base
from numpy import pi, reshape, split, vstack, zeros
from scipy.integrate import odeint
from copy import deepcopy
from numpy import array, cos, sin, linspace
from random import random
from numpy.linalg import norm

class PointLanding2D(base):
    # Mass optimal landing of thrust vectored point mass
    def __init__(
        self,
        s0     = [0.,1000.,20.,-5,10000.],
        st     = [0.,0.,0.,0.,9500.],
        c1     = 4400.,
        c2     = 311.*9.81,
        g      = 1.6229,
        nnodes = 5
    ):

        # NLP Dimension
        # Position, velocity, and mass
        self.SDim = 5     # x,y,vx,vy,m
        # Thrust magnitude and direction
        self.CDim  = 2     # T,theta
        # The initial and terminal states are given,
        # therefor the state will only be manipulated
        # at the interior transcription nodes.
        # With multiple shooting, controls are chosen
        # for every node except the last.
        self.NNodes   = nnodes
        self.NSeg     = self.NNodes - 1 # Ignore terminal
        self.NInNodes = self.NNodes - 2 # Ignore both
        self.NLPDim   = self.SDim*self.NInNodes + self.CDim*self.NSeg + 1
        # The resulting NLP desiscion vector will look like:
        # D = [s_1,..,s_N-1,u_0,..,u_N-1,tf]
        # Constraints will enforce that the system's state
        # propogated from node i will match that of node i+1
        self.NCons = self.SDim*self.NSeg
        # Initialise PyGMO problem (ESA-ACT)
        super(PointLanding2D,self).__init__(self.NLPDim,0,1,self.NCons,0,1e-5)

        # Store inputs
        self.s0 = s0
        self.st = st
        self.c1 = c1
        self.c2 = c2
        self.g  = g

        # State bounds
        self.SLB = [-500.,0.,-100.,-100.,0.]
        self.SUB = [500.,1000.,100.,100.,10000.]

        # Control bounds
        self.CLB = [0.,0.]
        self.CUB = [self.c1,2*pi]

        # Set PyGMO problem bounds
        self.set_bounds(
            self.SLB*self.NInNodes + self.CLB*self.NSeg + [0.],
            self.SUB*self.NInNodes + self.CUB*self.NSeg + [10000.]
        )

    def _objfun_impl(self,x):
        s,c,tf = self.DecodeNLP(x)
        t  = linspace(0,tf,self.NNodes)
        sf = self.Propogate(s[-1],c[-1],t[-2],t[-1])[-1]
        return (sf[-1],)

    def randx(self):
        lb = array(self.lb)
        ub = array(self.ub)
        return random()*(ub-lb)+lb

    def rands(self):
        lb = array(self.SLB)
        ub = array(self.SUB)
        return random()*(ub-lb)+lb

    def randc(self):
        lb = array(self.CLB)
        ub = array(self.CUB)
        return random()*(ub-lb)+lb

    def Dynamics(self, s, t, T, theta):
        x,y,vx,vy,m = s
        c1 = self.c1
        c2 = self.c2
        dx  = vx
        dy  = vy
        dvx = c1*T/m*sin(theta)
        dvy = c1*T/m*cos(theta)
        dm  = -c1*T/c2
        return [dx,dy,dvx,dvy,dm]

    def DecodeNLP(self,x):
        x    = array(x)
        s,x  = split(x,[self.SDim*self.NInNodes])
        c,tf = split(x,[self.CDim*self.NSeg])
        s    = s.reshape(self.NInNodes,self.SDim)
        c    = c.reshape(self.NSeg,self.CDim)
        tf   = tf[0]
        return s, c, tf

    def _compute_constraints_impl(self,x):
        s,c,tf = self.DecodeNLP(x)
        # Create time grid
        t = linspace(0,tf,self.NNodes)
        # Append initial and terminal states
        s = vstack((self.s0,s,self.st))
        # Create constraint vector
        ceq = []
        for n in range(self.NSeg):
            t0 = t[n]
            tf = t[n+1]
            s0 = s[n]
            sf = s[n+1]
            c0 = c[n]
            sfp = self.Propogate(s0,c0,t0,tf)[-1]
            ceq += list(sfp-sf)
        return ceq

    def Propogate(self,s0,c,t0,tf):
        T, theta = c
        t = linspace(t0,tf,2)
        s1 = odeint(self.Dynamics,s0,t,args=(T,theta))
        return s1


    def Nondim(self, s):
        snd = deepcopy(s)
        snd[0] /= self.R
        snd[1] /= self.R
        snd[2] /= self.V
        snd[3] /= self.V
        snd[4] /= self.M
        return snd

class Rocket2D(base):
    def __init__(self,Isp,Tmax,phimax=10)

if __name__ == '__main__':
