'''
Christopher Iliffe Sprague
christopher.iliffe.sprague@gmail.com

Low-thrust trajectory from parking orbit about
Earth to periodic orbit about L2-Lagrange point,
using multiple lunar flybys.
'''

from PyGMO.problem import base
from PyKEP.planet import spice
from PyKEP.util import load_spice_kernel
from PyKEP import MU_SUN, MU_EARTH

class LTL2(base):

    def __init__(
        self, # Obviously
        t0=[13000, 13100], # Launch window [mjd]
        tof=[0.5, 300], # Duration of leg [days]
        mass=[800, 2000], # [Dry mass, wet mass] [kg]
        Tmax=0.4, # Max thrust [N]
        Isp=2500, # Specific impulse [sec]
        tom=300, # Max mission durations [days]
    ):

        # Construct PyGMO problem.
        super(LTL2, self).__init__(
            6+nseg*3, # Dimensionality
            0,        # Integer dimensionality
            1,        # Objectives (final mass)
            8+nseg,   # Constraints
            1+nseg,   # Inequality constraints
            1e-4,     # Error tolerance
        )

        # Load ephemerides
        load_spice_kernel('de423.bsp')
        self.Primary   = spice('EARTH', 'EMB')
        self.Secondary = spice('HAYABUSA', 'EMB')



if __name__ == "__main__":
    Mission = LTL2()
    r1,v1 = Mission.Primary.eph(7564.33)
    r2,v2 = Mission.Secondary.eph(7564.33)
    print r1
    print v1
