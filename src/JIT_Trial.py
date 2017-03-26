''' -----------------------
Landing using JIT and PyGMO
Numba JIT is rather
restrictive in defining
classes, so functions are
defined extrenally. We use
here the Hermite-Simpson
seperated transcription.
------------------------'''
import numba
from numpy import *
from PyGMO.problem import base
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# NOTE: This didn't work, should just migrate finally migrate to C++

c1     = float64(44000)
c2     = float64(311.*9.81)
a      = float64(0)
CD     = float64(2.2)
rho    = float64(1.271)
A      = float64(0.01)
g0     = float64(9.81)
g      = g0
sdim   = 5
cdim   = 2
nnodes = 100
nsegs  = nnodes - 1
nlpdim = 1 + sdim + cdim + (sdim + cdim)*2*nsegs
condim = sdim*2 - 1 + sdim*2*nsegs
slb    = array([-3000, 0, -200, -200, 10], float64)
sub    = array([3000, 2000, 200, 200, 11000], float64)
tlb    = float64(1)
tub    = float64(200)
clb    = array([0, 0], float64)
cub    = array([1, 2*pi], float64)
nlplb  = hstack(([tlb], slb, clb))
nlpub  = hstack(([tub], sub, cub))
for i in range(nsegs):
    nlplb = hstack((nlplb, slb, clb, slb, clb))
    nlpub = hstack((nlpub, sub, cub, sub, cub))

@numba.jit(numba.float64[:](numba.float64[:], numba.float64[:]), nopython=True, cache=True, nogil=True)
def EOM_State(state, control):
    x, y, vx, vy, m = state
    u, theta        = control
    x0 = c1*u/m
    x1 = abs(vx)**2
    x2 = abs(vy)**2
    x3 = A*CD*rho*(x1/2. + x2/2.)/sqrt(x1 + x2)
    return array([
        vx,
        vy,
        -vx*x3 + x0*sin(theta),
        -g - vy*x3 + x0*cos(theta),
        -c1*u/c2
    ], float64)

@numba.jit(
    numba.types.Tuple((
        numba.float64,
        numba.float64[:,:],
        numba.float64[:,:],
        numba.float64[:,:],
        numba.float64[:,:]
    ))(numba.float64[:]), cache=True, nogil=True)
def HSS_Decode(z):
    tf   = float64(z[0])
    z    = array(z[1:], float64)
    b1   = zeros(sdim + cdim)
    z    = hstack((b1, z))
    z    = z.reshape((nnodes, (sdim + cdim)*2))
    i, j = 0, sdim
    sb   = z[:,i:j]
    i, j = j, j + cdim
    cb   = z[:,i:j]
    i, j = j, j + sdim
    s    = z[:,i:j]
    i, j = j, j + cdim
    c    = z[:,i:j]
    return tf, sb, cb, s, c

@numba.jit(numba.float64[:](numba.float64[:], numba.float64[:], numba.float64[:]), cache=True, nogil=True)
def HSS_Defects(z, si, st):
    tf, sb, cb, s, c = HSS_Decode(z)
    h = tf/nnodes
    # Boundary conditions
    ceq = s[0] - si
    ceq = hstack((ceq, s[-1,:-1] - st[:-1]))
    # Dynamic Constraints
    for k in range(nsegs):
        f1 = EOM_State(s[k], c[k])
        f2 = EOM_State(s[k+1], c[k+1])
        fb2 = EOM_State(sb[k+1], cb[k+1])
        # Hermite interpolation
        ceq1 = sb[k+1] - 0.5*(s[k+1] + s[k]) - h/8.*(f1-f2)
        # Simpson quadrature
        ceq2 = s[k+1] - s[k] - h/6.*(f2 + 4*fb2 + f1)
        ceq  = hstack((ceq, ceq1, ceq2))
    return ceq

@numba.jit(numba.float64[:](numba.float64), cache=True, nogil=True)
def Objective(z):
    tf, sb, cb, s, c = HSS_Decode(z)
    return -s[-1, -1]

@numba.jit(numba.float64[:](numba.float64[:], numba.float64))
def EOM(state, t):
    control = array([0, 0], float64)
    return EOM_State(state, control)
@numba.jit(numba.float64[:,:](numba.float64[:], numba.float64))
def Propagate(si, tf):
    return odeint(EOM, si, linspace(0, tf, nnodes), rtol=1e-12, atol=1e-12)

@numba.jit(numba.float64[:](numba.float64[:,:], numba.float64), nogil=True, cache=True)
def Code_Ballistic(states, tf):
    controls = zeros((nnodes, cdim))
    z = hstack((states, controls, states, controls))
    z = z.flatten()
    z = z[sdim+cdim:]
    z = hstack((tf, z))
    return z

class HSS_Trajectory(base):
    def __init__(
        self,
        si = array([0, 1000, 20, -5, 10000], float64),
        st = array([0, 0, 0, 0, 9000], float64)
    ):
        base.__init__(self, nlpdim, 0, 1, condim, 0, 1e-8)
        self.set_bounds(nlplb, nlpub)
        self.si = si
        self.st = st
    def _objfun_impl(self, z):
        return (Objective(z),)
    def _compute_constraints_impl(self, z):
        si, st = self.si, self.st
        return HSS_Defects(z, si, st)

if __name__ == "__main__":
    s  = array([0, 1000, 20, -5, 10000], float64)
    st = array([0, 0, 0, 0, 9000], float64)
    c  = array([0, 0], float64)
    tf = float64(20)
    z  = [tf] + list(s) + list(c) + (list(s) + list(c))*2*nsegs
    z  = array(z, float64)
    s  = Propagate(s, tf)
    z  = Code_Ballistic(s, tf)
    tf, sb, cb, s, c =  HSS_Decode(z)
    plt.plot(s[:,0], s[:,1], 'k.-')
    plt.show()
