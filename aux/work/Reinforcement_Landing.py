'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
// Reinforcement Learning Space Landing                            //
// Authored by Christopher Iliffe Sprague                          //
// Christopher.Iliffe.Sprague@gmail.com                            //
// +1 703 851 6842                                                 //
// https://github.com/CISprague/Astrodynamics_Machine_Learning.git //
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from   scipy            import   ndimage
from   numpy            import  sin, cos
import numpy                as        np
import matplotlib.pyplot    as       plt
import scipy.integrate      as integrate
import matplotlib.animation as animation
import matplotlib.image     as     image
import os

class Environment(object):
    instances = []

    def __init__(self,
                 name,
                 wind_angle=None,  # [degrees]
                 wind_speed=None):  # [m/s]
        self.name = name.strip().title()
        self.G = {'Earth': 9.807,
                  'Mars': 3.711,
                  'Moon': 1.622}[self.name]
        if wind_angle == None:
            self.wind_angle = np.random.uniform(np.pi, 3*np.pi/4.0)
        else:
            self.wind_angle = wind_angle

        if wind_speed == None:
            self.wind_speed = np.random.uniform(0.0, 30.0)
        else:
            self.wind_speed = wind_speed
        self.v_inf = np.array([self.wind_speed * cos(self.wind_angle),
                               self.wind_speed * sin(self.wind_angle)])
        Environment.instances.append(self)

    def Atmospheric_Density(self, h):
        h_vec=h
        h=np.linalg.norm(h)
        if self.name == 'Earth':
            # https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
            if h > 25000:
                T = -131.21 + 0.00299 * h_vec
                p = 2.488 * ((T + 273.1) / 216.6)**(-11.388)
            elif 11000 < h and h <= 25000:
                T = -56.46
                p = 22.65 * np.exp(1.73 - 0.000157 * h_vec)
            else:
                T = 15.04 - 0.00649 * h_vec
                p = 101.29 * ((T + 273.1) / 288.08)**(5.256)
            return p / (0.2869 * (T + 273.1))
        elif self.name == 'Mars':
            # https://www.grc.nasa.gov/www/k-12/airplane/atmosmrm.html
            if h > 7000:
                T = -23.4 - 0.00222 * h_vec
                p = 0.699 * np.exp(-0.00009 * h_vec)
            else:
                T = -31 - 0.000998 * h_vec
                p = 0.699 * np.exp(-0.00009 * h_vec)
            return p / (0.1921 * (T + 273.1))
        else:
            return 0.0

    def Report(self):
        for char in self.__dict__.keys():
            print char.replace('_', ' ').title() + ':', self.__dict__[char]

class Lander:
    """
    In SI units:
    S  = [x,  y,  vx, vy, theta, omega, m   ]
    S' = [vx, vy, ax, ay, omega, alpha, mdot]
    A  = [Tm, Ta, Tb]
    """

    def __init__(self,
                 env      = Environment('Mars'),
                 state    = [0.0, 1000.0, 40.0, -20.0, np.pi/2., 0.0, 300.0],
                 L        = 5.0,   # Length of lander [m]
                 W        = 1.0,   # Width of lander [m]
                 Isp      = 3000.0,
                 origin   = (0, 0)):
        self.env          = env # Environment of lander
        self.state        = np.asarray(state, dtype='float')
        self.params       = (L, W, Isp) # Length and width
        self.origin       = origin
        self.time_elapsed = 0
        self.states       = np.array(state, ndmin=2)
        self.times        = np.array([0])

    def dstate_dt(self, state, t):
        """compute the derivative of the given state"""
        x, y, vx, vy, theta, omega, mass = self.state
        (L, W, Isp), I = self.params, self.Inertia(mass)

        #Actions
        T          = 100.000 #[N]
        phi_g      = 0.174 #[rad]

        # Translational
        FG, FD, FT = self.Gravity(), self.Drag(), self.Thrust(T, phi_g)
        F          = FG + FD + FT
        ax, ay     = F / mass

        # Rotational
        Tau_T      = self.Thrust_Torque(T, phi_g) # + self.Drag_Torque()
        Tau        = Tau_T # + Tau_D
        alpha      = Tau / I

        #Propellant
        mdot       = T/(Isp*9.807)

        # State transition
        dydx       = np.zeros_like(state)
        dydx[0]    = state[2]
        dydx[1]    = state[3]
        dydx[2]    = ax # X-acceleration
        dydx[3]    = ay # y-acceleration
        dydx[4]    = state[5]
        dydx[5]    = alpha # Angular acceleration
        dydx[6]    = -mdot # Mass flow rate based on thrust
        return dydx

    def step(self, dt):
        """Takes one step in time."""
        self.state  = integrate.odeint(self.dstate_dt, self.state, [0, dt])[1]
        self.time_elapsed += dt
        self.times  = np.append(self.times, self.time_elapsed)
        self.states = np.append(self.states, [self.state], axis=0)

    def Inertia(self, mass):
        '''Moment of inertia for cylinder about central diameter.'''
        L, W, Isp = self.params # Length and width
        R    = W/2.0 # Radius
        return (mass/12.) * (3*R**2 + L**2)

    def Flow_Inclination(self):
        '''Inclination angle of lander w.r.t. wind direction.'''
        x, y, vx, vy, theta, omega, mass = self.state
        theta_inf = self.env.wind_angle
        return theta - theta_inf

    def Gravity(self, verbose=False):
        '''Downward gravitational force due to environment.'''
        mass = self.state[6]
        return self.env.G * np.array([0,-1]) * mass

    def Drag(self):
        '''Aerodynamic force felt by lander.'''
        x, y, vx, vy, theta, omega, mass = self.state
        v         = np.array([vx, vy])
        phi_inf   = self.Flow_Inclination()
        CD        = self.Drag_Coefficient(phi_inf)
        rho       = self.env.Atmospheric_Density(y)
        A_perp    = self.Perpendicular_Area(phi_inf)
        v_rel     = v - self.env.v_inf
        v_rel_mag = np.linalg.norm(v_rel)
        return -(0.5 * CD * A_perp * v_rel_mag * v_rel)

    def Drag_Torque(self):
        '''Torque imparted by atmospheric drag.'''
        x, y, vx, vy, theta, omega, mass = self.state
        L, W, Isp = self.params
        FD        = np.linalg.norm(self.Drag() * mass)
        phi_inf   = self.Flow_Inclination()
        I         = self.Inertia(mass)
        return FD*(cos(phi_inf)*L - sin(-phi_inf)*W)/I

    def Drag_Coefficient(self, phi_inf):
        '''Approximated drag coefficient for cylinder rotating about
        its cross-sectional axis: CD_min = 0.82, CD_max = 1.98.'''
        return 0.58*cos(2*phi_inf)+1.4

    def Perpendicular_Area(self, phi_inf):
        '''The area of the lander perpendicular to the flow.'''
        L, W, Isp = self.params
        return W*(abs(W*sin(phi_inf)) + abs(L*cos(phi_inf)))

    def Orientation(self):
        theta = self.state[4] # Inclination in main frame
        x_hat = np.array([cos(theta), sin(theta)])
        y_hat = np.array([-sin(theta), cos(theta)])
        return x_hat, y_hat

    def Thrust(self, T_mag, phi_gimbal):
        '''Force of thruster.'''
        x_hat, y_hat = self.Orientation()
        return T_mag * (cos(phi_gimbal)*y_hat - sin(phi_gimbal)*x_hat)

    def Thrust_Torque(self, T_mag, phi_gimbal):
        L, W, Isp = self.params
        x_hat, y_hat = self.Orientation()
        Tx = -T_mag * sin(phi_gimbal)
        return Tx * L

    def Aero_Plot(self):
        p        = np.linspace(0,4*np.pi, 5000)
        fig, ax1 = plt.subplots()
        ax2      = ax1.twinx()
        ax1.plot(np.degrees(p), Falcon.Drag_Coefficient(p), 'b')
        ax2.plot(np.degrees(p), Falcon.Perpendicular_Area(p), 'g')
        ax1.set_xlabel('$Inclination\,to\,Flow\: \phi_{\infty}\, [^\circ]$')
        ax1.set_ylabel('$Coefficient\, of\, Drag\: C_D$', color='b')
        ax2.set_ylabel('$Area\, Perpendicular\, to\, Flow\: A_{\perp \^\infty}\, [m^2]$', color='g')
        ax1.set_xlim([0,720])
        plt.show()



if __name__ == '__main__':
    # Initialisations
    Earth  = Environment('Earth')
    Falcon = Lander(Earth)
    print Falcon.Inertia(Falcon.state[-1])


'''
    dt = 1. / 30.  # 30 fps

    #///////////////////////////////////////////////////////////////////////////
    # set up figure and animation
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(0,1000), ylim=(0,1000))
    ax.grid()
    line,     = ax.plot([], 'o-', lw=2)
    status    = ax.text(0.02, 0.50, '', transform=ax.transAxes)

    def init():
        """initialize animation"""
        line.set_data([], [])
        status.set_text('')
        return line, status

    def animate(i):
        """perform animation step"""
        global Falcon, dt; Falcon.step(dt)
        x, y, vx, vy, theta, omega, mass = Falcon.state
        L, W, Isp = Falcon.params # Length and width of rocket
        line.set_data(x, y)
        t         = Falcon.time_elapsed
        aax, aay  = Falcon.Drag()/mass
        theta     = np.degrees(theta)%360
        v         = np.linalg.norm([vx, vy])
        status.set_text('$Time = %.2f$\n' % t +
                        '$a_D = (%.2f, %.2f)$\n' % (aax, aay) +
                        '$v = %.2f$\n' % v +
                        '$angle = %.2f$\n' % theta +
                        '$mass = %.4f$\n' % mass)
        return line, status

    # Interval based on dt and time to animate one step
    from time import time
    t0       = time()
    animate(0)
    t1       = time()
    interval = 1000 * dt - (t1 - t0)

    ani = animation.FuncAnimation(fig, animate, frames=300,
                                  interval=interval, blit=True, init_func=init)
    #ani.save('falcon9.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()
'''
