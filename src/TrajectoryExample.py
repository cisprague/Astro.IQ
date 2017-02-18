from PyGMO.problem        import base
from PyKEP                import MU_SUN, epoch, AU, EARTH_VELOCITY
from PyKEP.planet         import jpl_lp
from PyKEP.sims_flanagan  import spacecraft, leg, sc_state
from PyKEP.orbit_plots    import plot_planet, plot_sf_leg
from PyGMO                import algorithm, island
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt


class mga_lt_earth_mars(base):

    """
    This constructs a PaGMO.problem object that represents a low-thrust transfer between Earth and Mars. The decision vector
    contains [t0,T,mf,Vx,Vy,Vz,[throttles]] in the following units: [mjd2000, days, kg, m/s,m/s,m/s, [non-dimensional]]
    """

    def __init__(self, mass=1000, Tmax=0.05, Isp=2500, Vinf=3.0, nseg=20):
        # First we call the constructor for the base PyGMO problem
        # (dim, integer dim, number of obj, number of con, number of inequality con, tolerance on con violation)
        super(mga_lt_earth_mars, self).__init__(
            6 + nseg * 3, 0, 1, 8 + nseg, nseg + 1, 1e-4
        )

        self.earth = jpl_lp('earth')
        self.mars  = jpl_lp('mars')
        self.sc    = spacecraft(mass, Tmax, Isp)
        self.Vinf  = Vinf * 1000
        self.leg   = leg()
        self.leg.set_mu(MU_SUN)
        self.leg.set_spacecraft(self.sc)
        self.nseg  = nseg
        self.set_bounds(
            [2480, 2400, self.sc.mass / 10, -self.Vinf, -self.Vinf, -self.Vinf] + [-1]* 3 * nseg,
            [2490, 2500, self.sc.mass, self.Vinf, self.Vinf, self.Vinf] + [1] * 3 * nseg
        )

    # This is the objective function
    def _objfun_impl(self, x):
        return (-x[2],)

    # And these are the constraints
    def _compute_constraints_impl(self, x):
        start = epoch(x[0])
        end   = epoch(x[0] + x[1])

        r, v  = self.earth.eph(start)
        v     = [a + b for a, b in zip(v, x[3:6])]
        x0    = sc_state(r, v, self.sc.mass)

        r, v  = self.mars.eph(end)
        xe    = sc_state(r, v, x[2])

        self.leg.set(start, x0, x[-3 * self.nseg:], end, xe)
        v_inf_con = (x[3] * x[3] + x[4] * x[4] + x[5] * x[5] -
                     self.Vinf * self.Vinf) / (EARTH_VELOCITY * EARTH_VELOCITY)
        retval = list(self.leg.mismatch_constraints() +
            self.leg.throttles_constraints()) + [v_inf_con]

        # We then scale all constraints to non-dimensional values
        retval[0] /= AU
        retval[1] /= AU
        retval[2] /= AU
        retval[3] /= EARTH_VELOCITY
        retval[4] /= EARTH_VELOCITY
        retval[5] /= EARTH_VELOCITY
        retval[6] /= self.sc.mass
        return retval

    # This transforms the leg into a high fidelity one
    def high_fidelity(self, boolean):
        self.leg.high_fidelity = boolean

    # And this helps to visualize the trajectory
    def plot(self, x):
        # Making sure the leg corresponds to the requested chromosome
        self._compute_constraints_impl(x)
        start = epoch(x[0])
        end   = epoch(x[0] + x[1])

        # Plotting commands
        fig = plt.figure()
        axis = fig.gca(projection='3d')
        # The Sun
        axis.scatter([0], [0], [0], color='y')
        # The leg
        plot_sf_leg(self.leg, units=AU, N=10, ax=axis)
        # The planets
        plot_planet(
            self.earth, start, units=AU, legend=True, color=(0.8, 0.8, 1), ax = axis)
        plot_planet(
            self.mars, end, units=AU, legend=True, color=(0.8, 0.8, 1), ax = axis)
        plt.show()

def run_example1(n_restarts=5):

    prob                = mga_lt_earth_mars(nseg=15)
    prob.high_fidelity(True)
    algo                = algorithm.scipy_slsqp(max_iter=500, acc=1e-5)
    algo2               = algorithm.mbh(algo, n_restarts, 0.05)
    algo2.screen_output = True
    isl                 = island(algo2, prob, 1)
    print("Running Monotonic Basin Hopping .... this will take a while.")
    isl.evolve(1)
    isl.join()
    print("Is the solution found a feasible trajectory? " +
          str(prob.feasibility_x(isl.population.champion.x)))
    prob.plot(isl.population.champion.x)

if __name__ == "__main__":
    p = mga_lt_earth_mars()
    print p
