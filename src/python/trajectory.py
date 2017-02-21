from numpy import array, sqrt, hstack, linspace
from numpy.linalg import norm
from scipy.integrate import ode, odeint
import matplotlib.pyplot as plt
from PyGMO.problem import base, normalized
from PyGMO import *
import matplotlib.pyplot as plt

class PointLander2D(base):
    def __init__(
        self,
        s0 = array([0,1000,-5,0,10000],dtype=float),
        st = array([0,0,0,0,0],dtype=float),
        c1 = 44000.,
        c2 = 311.*9.81,
        g  = 1.6229,
        a = 0.9
    ):
        self.s0    = s0
        self.st    = st
        self.c1    = c1
        self.c2    = c2
        self.g     = g
        self.a     = a
        # PyGMO NLP: Z = [tf,lx,ly,lvx,lvy,lm]
        super(PointLander2D, self).__init__(6,0,1,6,0,1e-5)
        self.set_bounds([1]+[-1]*5, [400]+[1]*5)
        return None
    def EOM_State(self,state, control):
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
        dlvx = 0.
        dlvy = 0.
        dlm  = self.c1*u*(lvx*st+lvy*ct)/m**2
        return array([dlx,dly,dlvx,dlvy,dlm],dtype=float)
    def Pontryagin(self,fullstate):
        x,y,vx,vy,m,lx,ly,lvx,lvy,lm = fullstate
        lv = norm([lvx,lvy])
        st = -lvx/lv
        ct = -lvy/lv
        u = min(max((lm - self.a - lv*self.c2/m)/(2*self.c1*(1-self.a)),0),1)
        return array([u,st,ct],dtype=float)
    def EOM(self,t,fullstate):
        state = fullstate[:5]
        control = self.Pontryagin(fullstate)
        ds = self.EOM_State(state, control)
        dl = self.EOM_Costate(fullstate, control)
        return hstack((ds, dl))
    def Shoot(self,decision):
        tf = decision[0]
        l0 = decision[1:]
        fs0 = hstack((self.s0,l0))
        t = linspace(0,tf,500)
        sol = ode(self.EOM).set_integrator('dopri5')
        sol.set_initial_value(fs0,0)
        tl = []; fsl = []
        def Out(t,y): tl.append(t), fsl.append(y)
        sol.set_solout(Out)
        sol.integrate(tf)
        return array(tl), array(fsl)

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
    # PyGMO Stuff (Thanks ESA!)
    def _objfun_impl(self, decision):
        return (1.,)
    def _compute_constraints_impl(self, decision):
        t, fsf = self.Shoot(decision)
        fsf = fsf[-1]
        xf,yf,vxf,vyf,mf,lxf,lyf,lvxf,lvyf,lmf = fsf
        xt,yt,vxt,vyt,mt = self.st
        H = self.Hamiltonian(fsf)
        return [xf-xt,yf-yt,vxf-vxt,vyf-vyt,lmf,H]

if __name__ == '__main__':
    prob = PointLander2D()
    s = prob.s0
    c = array([0.5,0.5,0.5])
    l = array([1.1,1.3,1.2,1.2,1.5])
    fs = hstack((s,l))
    dec = array([100.0,0.1,0.1,0.1,0.1,0.1])
    #print prob.Shoot(dec)
    #sol = prob.Shoot(dec)
    #plt.plot(sol[:,0],sol[:,1])
    #plt.show()
    #print prob.Pontryagin(fs)
    normalized(prob)

    algo = algorithm.scipy_slsqp(max_iter = 10000,acc = 1E-8,epsilon = 1.49e-8, screen_output = True)
    algo.screen_output = True

    for i in range(1,1000):
        print("Attempt {}".format(i))
        pop = population(prob,1)
        pop = algo.evolve(pop)
        pop = algo.evolve(pop)
        if (prob.feasibility_x(pop[0].cur_x)):
            print(" - Success, violation norm is: {0:.4g}".format(norm(pop[0].cur_c)))
            break
        else:
            print(" - Failed, violation norm is: {0:.4g}".format(norm(pop[0].cur_c)))

    print("PaGMO reports: ")
    print(prob.feasibility_x(pop[0].cur_x))

    decf = array(pop[0].cur_x)
    sol = prob.Shoot(decf)
    plt.plot(sol[:,0],sol[:,1])
    plt.show()
    plt.plot(sol[:,2],sol[:3])
    plt.show()
