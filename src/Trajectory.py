from PyGMO.problem._base import base
from PyGMO.problem import normalized
from numpy import pi, reshape, split, vstack, zeros, float64, float32, hstack
from scipy.integrate import odeint
from copy import deepcopy
from numpy import array, cos, sin, linspace, concatenate, append, empty
from random import random
from numpy.linalg import norm
from PyGMO import *
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class PointLander2D(base):
    # Mass optimal landing of thrust vectored point mass
    # x = [tf,y1,u1,u2b,...,umb,ym,um]
    # y = [x,y,vx,vy,m], u = [T,phi]
    def __init__(
        self,
        initial_state    = array([0,1000,20,-5,10000],dtype=float),
        target_state     = array([0,0,0,0,9500],dtype=float),
        max_thrust       = 44000,
        specific_impulse = 311,
        gravity          = 1.6229,
        nnodes           = 5
    ):
        self.s0 = initial_state
        self.st = target_state
        self.c1 = max_thrust
        self.c2 = specific_impulse*9.81
        self.g  = gravity

        self.nnodes = nnodes
        self.nsegs  = self.nnodes-1
        self.sdim   = 5
        self.cdim   = 2
        self.fsdim  = self.sdim + self.cdim
        self.decdim = 1 + self.sdim + self.cdim + (self.sdim + self.cdim*2)*self.nsegs
        self.condim = self.sdim*self.nsegs + (self.sdim - 1)
        super(PointLander2D,self).__init__(self.decdim,0,1,self.condim,0,1e-5)
        self.slb = [-1000,0,-200,-400,400]
        self.sub = [1000,1000,200,400,10000]
        self.clb = [0,0]
        self.cub = [1,pi]
        self.fslb = self.slb + self.clb
        self.fsub = self.sub + self.cub
        self.nlplb = array([0] + self.fslb + (self.clb + self.fslb)*self.nsegs, dtype=float)
        self.nlpub = array([10000] + self.fsub + (self.cub + self.fsub)*self.nsegs,dtype=float)
        self.set_bounds(self.nlplb,self.nlpub)

    def samplex(self):
        return self.nlplb + (self.nlpub - self.nlplb)/2.

    def decodex(self, x):
        tf     = x[0]
        x      = concatenate(([0,0],x[1:]))
        x      = x.reshape(self.nnodes, self.fsdim + self.cdim)
        ub,s,u = split(x,[2,7],axis=1)
        return tf,ub,s,u

    def EOM(self, state, control):
        x,y,vx,vy,m = state
        u,phi   = control
        dx      = vx
        dy      = vy
        dvx,dvy = (self.c1*u/m)*array([sin(phi),cos(phi)]) + array([0,-self.g])
        dm      = -self.c1*u/self.c2
        return array([dx,dy,dvx,dvy,dm])

    def _compute_constraints_impl(self, x):
        tf,ub,y,u = self.decodex(x)
        t = linspace(0,tf,self.nnodes)
        ceq = empty([1,0])
        # Hermite-Simpson defects
        for k in range(self.nsegs):
            hk    = t[k+1] - t[k]
            fk    = self.EOM(y[k],u[k])
            fkp1  = self.EOM(y[k+1],u[k+1])
            ykp1b = 0.5*(y[k] + y[k+1]) + (hk/8.)*(fk - fkp1)
            fkp1b = self.EOM(ykp1b,ub[k+1])
            xi = y[k+1] - y[k] - (hk/6.)*(fk + 4*fkp1b +fkp1)
            ceq = append(ceq,xi)
        # Terminal condition
        ceq = append(ceq,self.st[:-1] - y[self.nsegs,:-1])
        return ceq

    def _objfun_impl(self,x):
        tf,ub,s,u = self.decodex(x)
        mf = s[-1,-1]
        return(-mf,)

    def plottraj(self,x):
        tf,ub,s,u = self.decodex(x)
        t = linspace(0,tf,self.nnodes)
        x = s[:,0]
        y = s[:,1]
        T = u[:,0]
        phi = u[:,1]
        plt.subplot(2,2,1)
        plt.plot(x,y)
        plt.subplot(2,2,2)
        plt.plot(t,T)
        plt.subplot(2,2,3)
        plt.plot(t,phi)
        plt.show()


def main():
    prob = PointLander2D(nnodes=30)
    #prob = normalized(L)
    algo = algorithm.scipy_slsqp(screen_output=True)
    algo.screen_ouput = True

    for i in range(15):
        print("Attempt # {}".format(i))
        pop = population(prob,1)
        pop = algo.evolve(pop)
        pop = algo.evolve(pop)
        if (prob.feasibility_x(pop[0].cur_x)):
            print("Success, violation norm is: {0:.4g}".format(norm(pop[0].cur_c)))
            break
        else:
            print("Failed, violation norm is: {0:.4g}".format(norm(pop[0].curc_c)))
    x = pop[0].cur_x
    prob.plottraj(x)

if __name__ == '__main__':
    main()
