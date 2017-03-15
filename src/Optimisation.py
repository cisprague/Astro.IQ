''' --------
Dependencies
-------- '''
from numpy import *
from numpy.linalg import *
from scipy.optimize import *
from scipy.interpolate import *
from Trajectory import PointLander
from PyGMO.problem import base
import matplotlib.pyplot as plt

''' ------------
Solution Guesses
------------ '''
class Guess(object):
    @staticmethod
    def Zeros(mesh):
        return zeros(len(mesh))
    @staticmethod
    def Constant(mesh, const):
        return ones(len(mesh))*const
    @staticmethod
    def Linear(mesh, yi, yt):
        x = array([mesh[0], mesh[-1]])
        y = array([yi, yt])
        z = interp1d(x, y)
        return z(mesh)
    @staticmethod
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

''' ----------------------
Direct Collocation Methods
---------------------- '''
class Trapezoidal(base):
    def __init__(self, model=PointLander(), nsegs=20):
        self.nsegs  = nsegs
        self.nnodes = nsegs + 1
        self.model  = model
        dim  = 1 + (model.sdim + model.cdim)*self.nnodes
        cdim = model.sdim*nsegs + 2*model.sdim - 1
        base.__init__(self, dim, 0, 1, cdim, 0, 1e-8)
        self.set_bounds(model.tlb + (model.slb + model.clb)*self.nnodes,
                        model.tub + (model.sub + model.cub)*self.nnodes)
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


if __name__ == "__main__":
    mesh = linspace(1, 20, 30)
    plt.plot(Guess.Cubic(mesh, 20, 5, -10, -5))
    plt.show()
