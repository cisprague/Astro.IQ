from numpy import (hsplit, array,zeros_like,insert,cross,append,radians,sin,cos,empty,pi,vstack,float64,linspace,empty,zeros,arange)
from numpy.linalg import (norm,inv)
from scipy.integrate import odeint
from PyGMO.problem._base import base
from numpy.random import randn, uniform
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

class Rocket(object):

    def __init__(
        self,
        state0 = [0,0,1000,5,0,-5,1,0,0,0,0,0,0,10000],
        statet = [0,0,0,0,0,0,1,0,0,0,0,0,0,None],
        params = [5,1,2000]
    ):
        '''
        Initial State: state0 | Target State: statet
        state = [x,y,z,vx,vy,vz,qr,qi,qj,qk,wx,wy,wz,m]
        in [m,m,m,m/s,m/s,m/s,ND,ND,ND,ND,rad/s,rad/s,rad/s,kg]
        Parametres: params = [L,W,Isp] in [m,m,sec]
        Optimization Parametres = [N_Nodes]

        Control Vector:
        Control vector for the spacecraft's propulsion
        system, control = [T,incl,azim]. T denotes the
        magnitude to which the thruster is fired, incl
        indicates the thrust vector's inclination angle
        measured from the positive z-axis in the body
        frame, and azim indicated the azimuthal angle of
        the thrust vector measured from the positive x-axis.
        These three spherical coordinate parametres form a
        control vector in the body fixed frame, from which
        the translation force vector is computed in the
        inertial frame, determined by the body's orientation.

        NLP Descision Vector:
        x = [s1,u1,s2,u2,...,sf,uf,t0,tf]
        x = [x1,y1,z1,vx1,vy1,vz1,qr1,qi1,qj1,qk1,...
             wx1,wy1,wz1,m1,T1,i1,a1,x2,y2,z2,vx2,...
             vy2,vz2,qr2,qi2,qj2,qk2,wx2,wy2,wz2,m2,...
             T2,i2,a2,...,xf,yf,zf,vxf,vyf,vzf,qrf,...
             qif,qjf,qkf,wxf,wyf,wzf,mf,Tf,if,af,tf]

        '''

        # Initialize
        self.State = array(state0,dtype =float64)
        self.Target = array(statet,dtype =float64)
        self.Params = array(params,dtype =float64)

        # Bounds
        self.StateBounds = array([
            [-1000,1000],[-1000,1000],[0,1000],
            [-100,100],[-100,100],[-100,100],
            [0,1],[0,1],[0,1],[0,1],
            [-10,10],[-10,10],[-10,10],
            [1000,1000]
        ])
        self.ContBounds = array([[0,1765800],[0,pi],[0,2*pi]],dtype=float64)

        # Nonlinear Programme
        self.StateDim = 14 # State dimensions
        self.ContDim  = 3  # Control dimensions
        self.NNodes   = 10 # Number of nodes
        self.NAux     = 1  # Number of auxilary variables
        self.NMain    = self.StateDim + self.ContDim
        self.NLPDim   = self.NMain*self.NNodes+self.NAux

    def Optimise(self,x0):
        cons = ({'type': 'eq', 'fun': self.Defects})
        return None

    def Bounds(self,x):
        print x


    def Objective(self,x):
        states,controls,tf = self.Decode_NLP(x)
        obj = -states[-1,-1] # Maximize mass
        return obj

    def Constraints(self,x):
        '''
        Returns scaler objective and equality constraints.
        '''
        states,controls,tf = self.Decode_NLP(x)
        t = linspace(0,tf,self.NNodes) # Time grid
        constr = states[0]-self.State # Initial
        print constr
        for n in range(self.NNodes-1):
            s1 = states[n]
            s2 = states[n+1]
            u1 = controls[n]
            u2 = controls[n+1]
            h1 = t[n+1]-t[n]
            f1 = self.RHS(s1,u1)
            f2 = self.RHS(s2,u2)
            constr = append(constr,s2-s1-0.5*h1*(f1+f2))
        eqconstr = append(constr,states[-1]-self.Target) # Terminal
        return eqconstr

    def RHS(self,state,control):
        '''
        Returns rate of change of state
        given current state and control.
        Control:
        Thrust magnitude        = Tmag
        Body thrust inclination = incl
        Body thrust azimuth     = azim
        '''
        # State
        p = state[0:3]   # Position
        v = state[3:6]   # Velocity
        q = state[6:10]  # Quaternion
        w = state[10:13] # Euler Rates
        m = state[13]    # Mass
        # Parametres
        L,W,Isp = self.Params
        # Control parametres
        T, incl, azim = control
        # Thrust direction in body frame
        gamma = array([sin(incl)*cos(azim),
                       sin(incl)*sin(azim),
                       cos(incl)])
        # Thrust vector in body frame [N]
        ub  = T*gamma
        # Thrust vector in inertial frame [N]
        u   = self.QuatVecTrans(q,ub)
        # Moment arm of thrust [m]
        r   = array([0,0,-L/2])
        # Moment due to thrust [Nm]
        tau = cross(r,ub)
        # Inertia tensor
        I   = self.Cuboid_Inertia(W,W,L,m)
        # Gravity
        g = array([0,0,-9.807])
        # Assemble first order system
        dS        = zeros_like(state)
        dS[0:3]   = v
        dS[3:6]   = g + u/m
        dS[6:10]  = self.dqdt(q,w)
        dS[10:13] = tau.dot(inv(I))
        dS[13]    = -T/(Isp*9.807)
        return dS

    def Decode_NLP(self,x):
        '''
        Decodes the nonlinear programme descision
        vector.
        '''
        tf = x[-1]
        x  = x[:-1].reshape((self.NNodes,self.NMain))
        states, controls = hsplit(x,[self.StateDim])
        return states,controls,tf

    def xrand(self):
        xrand = []
        for n in range(self.NNodes):
            for s in range(self.StateDim):
                xrand.append(uniform(*self.StateBounds[s]))
            for u in range(self.ContDim):
                xrand.append(uniform(*self.ContBounds[u]))
        xrand.append(1000)
        return array(xrand,dtype=float64)

    @staticmethod
    def Eul2Quat(psi, theta, phi, unit):
        if unit is 'deg':
            psi,theta,phi = radians((psi,theta,phi))
        else: pass
        a, b, c = phi/2.0, theta/2.0, psi/2.0
        sa, ca  = sin(a), cos(a)
        sb, cb  = sin(b), cos(b)
        sc, cc  = sin(c), cos(c)
        qi      = sa*cb*cc - ca*sb*sc
        qj      = ca*sb*cc + sa*cb*sc
        qk      = ca*cb*sc - sa*sb*cc
        qr      = ca*cb*cc + sa*sb*sc
        return  array([qr,qi,qj,qk])

    @staticmethod
    def Quat2Eul(qi,qj,qk,qr,unit):
        phi   = arctan2(2*(qr*qi+qj*qk), 1-2*(qi**2+qj**2))
        theta = arcsin(2*(qr*qj-qk*qi))
        psi   = arctan2(2*(qr*qk+qi*qj), 1-2*(qj**2+qk**2))
        if unit is 'deg': return degrees((psi,theta,phi))
        else: return psi, theta, phi

    @staticmethod
    def Sphere2Cart(rho,theta,phi,unit='deg'):
        if unit is 'deg':
            theta,phi = radians((theta,phi))
        else: pass
        x = rho*sin(theta)*cos(phi)
        y = rho*sin(theta)*sin(phi)
        z = rho*cos(theta)
        return array([x,y,z])

    @staticmethod
    def QuatVecTrans(q,v):
        qr,qi,qk,qj = q
        qconj = array([qr,-qi,-qj,-qk])
        qnorm = norm(q)
        qinv  = qconj/qnorm
        v     = insert(v,0,0)
        v     = Rocket.QuatMult(v,qinv)
        v     = Rocket.QuatMult(q,v)
        return v[1:]

    @staticmethod
    def QuatMult(q1,q2):
        s1 = q1[0]
        v1 = q1[1:]
        s2 = q2[0]
        v2 = q2[1:]
        s  = s1*s2 - v1.dot(v2)
        v  = s1*v2 + s2*v1 + cross(v1,v2)
        return append(s,v)

    @staticmethod
    def dqdt(q,w):
        w = insert(w,0,0)
        return Rocket.QuatMult(0.5*q,w)

    @staticmethod
    def Cuboid_Inertia(w,h,d,m):
        Ix = (1/12.)*(h**2+d**2)
        Iy = (1/12.)*(w**2+d**2)
        Iz = (1/12.)*(w**2+h**2)
        I  = array([[Ix,0,0],[0,Iy,0],[0,0,Iz]])
        return I

    @staticmethod
    def Quat2RotMat(q):
        qr     = q[0]
        qi     = q[1]
        qj     = q[2]
        qk     = q[3]
        A      = empty([3,3])
        A[0,0] = qr**2+qi**2-qj**2-qk**2
        A[1,0] = 2*(qi*qj+qk*qr)
        A[2,0] = 2*(qi*qk-qj*qr)
        A[0,1] = 2*(qi*qj-qk*qr)
        A[1,1] = qr**2-qi**2+qj**2-qk**2
        A[2,1] = 2*(qj*qk+qi*qr)
        A[0,2] = 2*(qi*qk+qj*qr)
        A[1,2] = 2*(qj*qk-qi*qr)
        A[2,2] = qr**2-qi**2-qj**2+qk**2
        return A

if __name__ == "__main__":
    Falcon = Rocket()
    x = Falcon.xrand()
    Falcon.Constraints(x)
