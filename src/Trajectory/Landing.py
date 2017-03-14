from numpy import *

'''
class AlphaPointLander(object):
    def __init__(self,
        si   = [0, 1000, 20, -5, 10000],
        st   = [0, 0, 0, 0, 9000],
        Isp  = 311,
        g    = 1.6229,
        a    = 0,
        pin  = False,
        T    = 44000,
        nseg = 20,
        meth = 'HS',
        obj  = 'mass'
    ):
        self.si   = array(si, float)
        self.st   = array(st, float)
        self.Isp  = float(Isp)
        self.g    = float(g)
        self.a    = float(a)
        self.g0   = float(9.81)
        self.pin  = bool(pin)
        self.T    = float(T)
        self.nseg = int(nseg)
        self.meth = meth
        self.obj  = obj
        self.slb  = [-500, 0, -200, -200, 0]
        self.sub  = [500, 1000, 200, 200, 10000]
        self.clb  = [0, -1, -1]
        self.cub  = [1, 1, 1]
        self.tlb  = [1]
        self.tub  = [800]
        self.sdim = len(self.slb)
        self.cdim = len(self.clb)
        if self.meth is 'HS':
            self.Hermite_Simpson()
            self._compute_constraints_impl = self.Hermite_Simpson_Defects
        elif self.meth is 'indirect':
            self.Indirect()
            self._compute_constraints_impl = self.Indirect_Constraints
        elif self.meth is 'trap':
            self.Trapezoidal()
            self._compute_constraints_impl = self.Trapezoidal_Constraints
        else:
            pass
        if self.obj is 'mass':
            self._objfun_impl = self.Mass_Objective
        return None
    def Hermite_Simpson(self):
        slb, sub = self.slb, self.sub
        clb, cub = self.clb, self.cub
        tlb, tub = self.tlb, self.tub
        nseg     = self.nseg
        lb       = array(tlb + slb + clb + (clb + slb + clb)*nseg, float)
        ub       = array(tub + sub + cub + (cub + sub + cub)*nseg, float)
        dim      = len(lb)
        super(PointLander, self).__init__(
            dim, 0, 1, self.nseg*(self.sdim - 1) + self.sdim + self.sdim-1, 0, 1e-8
        )
        self.set_bounds(lb, ub)
        return None
    def Hermite_Simpson_Defects(self, decision):
        ub, y, u, tf = self.Decode_NLP(decision, 'HS')
        t            = linspace(0, tf, self.nseg + 1)
        # Boundary constraints
        ceq = list(y[0] - self.si) + list(y[-1, :-1] - self.st[:-1])
        # Dynamic constraints
        for k in range(self.nseg):
            tk    = t[k]
            tkp1  = t[k+1]
            hk    = tkp1 - tk
            yk    = y[k]
            ykp1  = y[k+1]
            uk    = u[k]
            ukp1  = u[k+1]
            fk    = self.EOM_State(yk, uk)
            fkp1  = self.EOM_State(ykp1, ukp1)
            ykp1b = 0.5*(yk + ykp1) + hk/8.*(fk - fkp1)
            ukp1b = ub[k+1]
            fkp1b = self.EOM_State(ykp1b, ukp1b)
            dk    = hk/6.*(fk + 4*fkp1b + fkp1)
            dk    = ykp1[:-1] - yk[:-1] - dk[:-1]
            ceq  += list(dk)
        return ceq
    def Mass_Objective(self, decision):
        ub, y, u, tf = self.Decode_NLP(decision)
        return (-y[-1, -1],)
    def Decode_NLP(self, decision, type='HS'):
        decision = list(decision)
        tf = decision.pop(0)
        if type is 'HS':
            decision = [0]*self.cdim + decision
            decision = array(decision, float)
            decision = decision.reshape((self.nseg+1, self.cdim*2 + self.sdim))
            ubar     = decision[:, 0:3]
            s        = decision[:, 3:8]
            u        = decision[:, 8:11]
            return ubar, s, u, tf
        else:
            return None
    def EOM_State(self, state, control, jac=False):
        x, y, vx, vy, m = state
        u, st, ct       = control
        x0 = self.T*u/m
        ds = array([
            vx,
            vy,
            st*x0,
            ct*x0 - self.g,
            -self.T*u/(self.Isp*self.g0)
        ], float)
        if jac:
            dds = array([
                [0, 0, 1, 0,        0],
                [0, 0, 0, 1,        0],
                [0, 0, 0, 0, -st*x0/m],
                [0, 0, 0, 0, -ct*x0/m],
                [0, 0, 0, 0,        0]
            ], float)
            return dds
        else:
            return ds
    def PlotTraj(self, decision):
        ub, y, u, tf = self.Decode_NLP(decision)
        plt.plot(y[:,0], y[:,1], 'k.-')
        plt.show()
'''
class PointLander(object):
    def __init__(
        self,
        si  = [10, 1000, 20, -5, 10000],
        st  = [0, 0, 0, 0, 9000],
        Isp = 311,
        g   = 1.6229,
        T   = 44000
    ):
        self.si   = array(si, float)
        self.st   = array(st, float)
        self.Isp  = float(Isp)
        self.g    = float(g)
        self.T    = float(T)
        self.sdim = 5
        self.cdim = 3
        self.slb  = [-500, 0, -200, -200, 0]
        self.sub  = [500, 1000, 200, 200, 10000]
        self.clb  = [0, -1, -1]
        self.cub  = [1, 1, 1]
        self.tlb  = [1]
        self.tub  = [800]
    def Optimise(self, method='trap', nsegs=20):
        if method is 'trap':
            Direct.Trapezoidal(self, nsegs)
        elif method is 'RK':
            Direct.Runge_Kutta(self, nsegs)
        elif method is 'HS':
            Direct.Hermite_Simpson(self, nsegs)
