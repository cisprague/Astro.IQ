from Rocket import Rocket
from numpy import pi

# Instantiate rocket
Falcon = Rocket()
S = Falcon.State[-1]
C = [5,pi/4,pi/2]
Falcon.State_Transition(S,C)
