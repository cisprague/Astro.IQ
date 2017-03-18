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
from numpy.linalg import *
from scipy.interpolate import *
from Trajectory import Point_Lander
from PyGMO.problem import base
import matplotlib.pyplot as plt

''' ------------
Solution Guesses
------------ '''
class Guess(object):
    def __init__(self, prob):
        self.prob = prob
    def Mid(self, nlp=True):
        z = self.prob.dlb + 0.5*(self.prob.dub - self.prob.dlb)
        if nlp: return z
        else: return self.prob.Decode(z)
    def Linear(self, lb, ub):
        mesh = linspace(0, 1, self.prob.nnodes)
        x    = array([mesh[0], mesh[-1]])
        a    = []
        for yi, yt in zip(lb, ub):
            y = array([yi, yt])
            z = interp1d(x, y)
            a.append(z(mesh))
        return transpose(array(a))
    def Cubic(mesh, yi, dyi, yt, dyt):
        ni, nt = mesh[0], mesh[-1]
        A = array([
            [1, ni, ni**2,   ni**3],
            [0,  1,  2*ni, 3*ni**2],
            [1, nt, nt**2,   nt**3],
            [0,  1,  2*nt, 3*nt**2]
        ])
        c = inv(A).dot(array([yi, dyi, yt, dyt]))
        return c[0] + c[1]*mesh + c[2]*mesh**2 + c[3]*mesh**3
    def Ballistic(self, si=None, tf=None, nlp=True):
        if si is None: si = self.prob.model.si
        if tf is None: tf = self.prob.model.tub
        s = self.prob.model.Propagate.Ballistic(
            si=si, tf=tf, nnodes=self.prob.nnodes
        )
        c = zeros((self.prob.nnodes, self.prob.model.cdim))
        if nlp:
            if self.prob.nc == 1: # Only node control
                return self.prob.Code(tf, s, c)
            elif self.prob.nc == 2: # Node and midpoint control
                return self.prob.Code(tf, c, s, c)
        else:
            return tf, s

''' ----------------------
Direct Collocation Methods
---------------------- '''
class Direct(object):
    def __init__(self, model=Point_Lander(), nsegs=20, ns=1, nc=1):
        self.nsegs  = int(nsegs)  # Number of segments
        self.nnodes = nsegs + 1   # Number of nodes
        self.ns     = ns          # Number of states per segement
        self.nc     = nc          # Number of controls per segement
        self.model  = model       # The dynamical model
        self.Guess  = Guess(self) # Guessing methods
class S1C1(Direct):
    def __init__(self, model=Point_Lander(), nsegs=20, ns=1, nc=1):
        Direct.__init__(self, model, nsegs, ns, nc)
        self.dim  = 1 + (model.sdim + model.cdim)*self.nnodes
        self.nc   = 1
        self.consdim = model.sdim*nsegs + 2*model.sdim - 1
        self.nobj = 1
        self.dlb  = array(
            [model.tlb] + (list(model.slb) + list(model.clb))*self.nnodes
        , float)
        self.dub  = array(
            [model.tub] + (list(model.sub) + list(model.cub))*self.nnodes
        , float)
        base.__init__(self, self.dim, 0, self.nobj, self.consdim, 0, 1e-8)
        self.set_bounds(self.dlb, self.dub)
    def _objfun_impl(self, z):
        return None
    def _compute_constraints_impl(self, z):
        return None
    def Decode(self, z):
        tf = z[0]
        z  = array(z[1:])
        z  = z.reshape((self.nnodes, self.model.sdim + self.model.cdim))
        s  = z[:, 0:self.model.sdim]
        c  = z[:, self.model.sdim:self.model.sdim+self.model.cdim]
        return tf, s, c
    def Code(self, tf, s, c):
        z = hstack((s, c))
        z = list(z.reshape((self.model.sdim + self.model.cdim)*self.nnodes))
        z = array([tf] + z)
        return z
class S1C2(Direct):
    def __init__(self, model=Point_Lander(), nsegs=20, ns=1, nc=1):
        Direct.__init__(self, model, nsegs, ns, nc)
        self.dim = 1 + model.sdim + model.cdim + (model.cdim*2 + model.sdim)*nsegs
        self.nc = 2 # Controls per segment
        self.consdim = model.sdim*nsegs + 2*model.sdim - 1
        self.dlb = array(
            [model.tlb] + list(model.slb) + list(model.clb) + (list(model.clb) + list(model.slb) + list(model.clb))*nsegs
        , float)
        self.dub = array(
            [model.tub] + list(model.sub) + list(model.cub) + (list(model.cub) + list(model.sub) + list(model.cub))*nsegs
        , float)
        base.__init__(self, self.dim, 0, 1, self.consdim, 0, 1e-8)
        self.set_bounds(self.dlb, self.dub)
    def _objfun_impl(self, z):
        return None
    def _compute_constraints_impl(self, z):
        return None
    def Decode(self, z):
        tf = z[0]
        z  = hstack((zeros(self.model.cdim), z[1:]))
        z  = z.reshape((self.nnodes, self.model.cdim*2 + self.model.sdim))
        cb = z[:,0:3]  # Midpoint control
        s  = z[:,3:8]  # States
        c  = z[:,8:11] # Nodal control
        return tf, cb, s, c
    def Code(self, tf, cb, s, c):
        z = hstack((cb, s, c))
        z = z.reshape((self.model.cdim*2 + self.model.sdim)*self.nnodes)
        z = list(z[self.model.cdim:]) # Remove midpoint control placeholder
        return array([tf] + z)
class Euler(S1C1, base):
    def __init__(self, model=Point_Lander(), nsegs=20, ns=1, nc=1):
        S1C1.__init__(self, model, nsegs, ns, nc)
    def _objfun_impl(self, z):
        tf, s, c = self.Decode(z)
        return (s[-1,-1],)
    def _compute_constraints_impl(self, z):
        tf, s, c = self.Decode(z)
        h        = tf/self.nnodes
        # Boundary
        ceq  = list(s[0] - self.model.si)
        ceq += list(s[-1,:-1] - self.st[:-1])
        # Dynamics
        for k in range(self.nnodes):
            ceq += list(s[k+1] - s[k] - h*self.model.EOM_State(s[k], u[k]))
        return ceq
class Trapezoidal(S1C1, base):
    def __init__(self, model=Point_Lander(), nsegs=20, ns=1, nc=1):
        S1C1.__init__(self, model, nsegs, ns, nc)
    def _objfun_impl(self, z):
        tf, s, c = self.Decode(z)
        return (-s[-1, -1],)
    def _compute_constraints_impl(self, z):
        tf, s, c = self.Decode(z)
        h        = tf/self.nnodes
        # Boundary
        ceq  = list(s[0] - self.model.si)
        ceq += list(s[-1,:-1] - self.model.st[:-1])
        # Dynamics
        for k in range(self.nsegs):
            f1 = self.model.EOM_State(s[k], c[k])
            f2 = self.model.EOM_State(s[k+1], c[k+1])
            ceq += list(s[k+1] - s[k] - h/2.*(f1 + f2))
        return ceq
class Runge_Kutta(S1C2, base):
    def __init__(self, model=Point_Lander(), nsegs=20, ns=1, nc=2):
        S1C2.__init__(self, model, nsegs, ns, nc)
    def _objfun_impl(self, z):
        tf, cb, s, c = self.Decode(z)
        return(-s[-1,-1],)
    def _compute_constraints_impl(self, z):
        tf, cb, s, c = self.Decode(z)
        h            = tf/self.nnodes
        # Boundary
        ceq  = list(s[0] - self.model.si) # Mass must match
        ceq += list(s[-1,:-1] - self.model.st[:-1]) # Mass free
        # Dyanamics
        for k in range(self.nsegs):
            k1 = h*self.model.EOM_State(s[k], c[k])
            k2 = h*self.model.EOM_State(s[k] + 0.5*k1, cb[k+1])
            k3 = h*self.model.EOM_State(s[k] + 0.5*k2, cb[k+1])
            k4 = h*self.model.EOM_State(s[k] + k3, c[k+1])
            ceq += list(s[k+1] - s[k] - 1/6.*(k1 + 2*k2 + 2*k3 + k4))
        return ceq
class Hermite_Simpson(S1C2, base):
    def __init__(self, model=Point_Lander(), nsegs=20):
        S1C2.__init__(self, model, nsegs, 1, 2)
    def _objfun_impl(self, z):
        tf, cb, s, c = self.Decode(z)
        return (-s[-1,-1],)
    def _compute_constraints_impl(self, z):
        tf, cb, s, c = self.Decode(z)
        h            = tf/self.nnodes
        # Boundary
        ceq  = list(s[0] - self.model.si)
        ceq += list(s[-1,:-1] - self.model.st[:-1])
        # Dyanmics
        for k in range(self.nsegs):
            f1 = self.model.EOM_State(s[k], c[k])
            f2 = self.model.EOM_State(s[k+1], c[k+1])
            sb2 = 0.5*(s[k] + s[k+1]) + h/8.*(f1 + f2)
            fb2 = self.model.EOM_State(sb2, cb[k+1])
            ceq += list(s[k+1] - s[k] -h/6.*(f1 + 4*fb2 + f2))
        return ceq

''' -----------
Indirect Method
----------- '''
class Indirect_Shooting(base):
    def __init__(self, model=Point_Lander()):
        self.model = model
        self.dim   = model.sdim     # Just guess costates
        self.cdim  = model.sdim + 1 # Infinite horizon



if __name__ == "__main__":
    mod = Point_Lander()
    prob = Indirect_Shooting(mod)
