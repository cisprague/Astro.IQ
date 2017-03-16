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
        if nlp: return self.prob.Code(tf, s, c)
        else: return tf, s, c

''' ----------------------
Direct Collocation Methods
---------------------- '''
class Direct(object):
    def __init__(self, model=Point_Lander(), nsegs=20):
        self.nsegs  = int(nsegs)
        self.nnodes = nsegs + 1
        self.model  = model
        self.Guess  = Guess(self)

class Trapezoidal(Direct, base):
    def __init__(self, model=Point_Lander(), nsegs=20):
        Direct.__init__(self, model, nsegs)
        self.dim  = 1 + (model.sdim + model.cdim)*self.nnodes
        self.cdim = model.sdim*nsegs + 2*model.sdim - 1
        self.nobj = 1
        self.dlb  = array(
            [model.tlb] + (list(model.slb) + list(model.clb))*self.nnodes
        , float)
        self.dub  = array(
            [model.tub] + (list(model.sub) + list(model.cub))*self.nnodes
        , float)
        base.__init__(self, self.dim, 0, self.nobj, self.cdim, 0, 1e-8)
        self.set_bounds(self.dlb, self.dub)
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

if __name__ == "__main__":
    apollo = Point_Lander()
    problem = Trapezoidal(apollo)
    tf, s, c = problem.Guess.Ballistic(tf=35, nlp=False)
    plt.plot(s[:,0], s[:,1])
    plt.show()
