from PyGMO.problem import base
from PyGMO import *
from numpy import *
from types import MethodType

class Trapezoidal(object):
    def __init__(self, problem, nsegs):
        self.nsegs   = nsegs
        self.nnodes  = nsegs + 1
        self.problem = problem
        dim  = 1 + (problem.sdim + problem.cdim)*self.nnodes
        cdim = problem.sdim*nsegs + 2*problem.sdim - 1
        problem.base.__init__(problem, dim, 0, 1, cdim, 0, 1e-8)
        problem.set_bounds(
            problem.tlb + (problem.slb + problem.clb)*self.nnodes,
            problem.tub + (problem.sub + problem.cub)*self.nnodes
        )
        problem._objfun_impl              = self.Objective
        problem._compute_constraints_impl = self.Constraints
    def Objective(self, z):
        tf, s, c = self.Decode(z)
        return (s[-1, -1],)
    def Constraints(self, z):
        tf, s, c = self.Decode(z)
        h        = tf/self.nnodes
        # Boundary
        ceq  = list(s[0] - self.problem.si)
        ceq += list(s[-1,:-1] - self.problem.st[:-1])
        # Dynamics
        for k in range(self.nsegs):
            f1 = self.problem.EOM_State(s[k], c[k])
            f2 = self.problem.EOM_State(s[k+1], c[k+1])
            ceq += list(s[k+1] - s[k] - h/2.*(f1 + f2))
        return ceq
    def Decode(self, z):
        tf = z[0]
        z  = array(z[1:])
        z  = z.reshape((self.nnodes, self.problem.sdim + self.problem.cdim))
        s  = z[:, 0:self.problem.sdim]
        c  = z[:, self.problem.sdim:self.problem.sdim+self.problem.cdim]
        return tf, s, c

if __name__ == "__main__":
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from Trajectory.Landing import *
    Apollo = PointLander()
    Apollo.Transcribe()
    pop = population(Apollo, 1)
