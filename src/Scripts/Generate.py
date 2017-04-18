# Resources
import sys
sys.path.append('../')
from Trajectory import Point_Lander_Drag
from Optimisation import HSS
from PyGMO import *
from numpy import *
from IQ import Random_Initial_States
import multiprocessing


''' -----------------
>> python Generate.py
----------------- '''

def initial_states():
    model = Point_Lander_Drag()
    print Random_Initial_States(model)
def trajectories():
    ''' -------------------------------------------------
    Problem       : SpaceX Dragon 2 Martian Soft Landing
    Dynamics      : 2-Dimensional Variable Mass Point
    Transcription : Hermite Simpson Seperated (HSS)
    Produces      : Database of fuel-optimal trajectories
    ------------------------------------------------- '''

    # Load initial states
    print("Loading initial states...")
    si_list = load('Data/Point_Lander_Mars_Initial_States.npy')

    # Define the algorithms to use
    algo_local = algorithm.scipy_slsqp(max_iter=5000, screen_output=True)
    algo_meta  = algorithm.mbh(algo_local, stop=1, screen_output=True)

    # Load initial guess
    print("Loading initial guess..")
    z = load('Data/HSS_10_Mars_Base.npy')

    # Allott space for solutions
    n_traj = len(si_list)
    sols   = zeros((n_traj, len(z)))
    # NOTE: Add cosine similarity in the loop to compare initial states
    # For each initial state
    for i in range(n_traj):
        si      = si_list[i]
        print("Trajectory " + str(i))
        print("State: " + str(si))
        # Initialise the model at that state
        model   = Point_Lander_Drag(si)
        # Initialise the HSS problem
        prob    = HSS(model, nsegs=10)
        # Create empty population
        pop     = population(prob)
        # Guess the previous solution
        pop.push_back(z)
        # Optimise from that solution
        print("Beginning optimisation...")
        pop     = algo_local.evolve(pop)
        # Store the new solution
        z       = pop.champion.x
        # Save the solution
        save("Data/Mars/HSS_10A_" + str(i), z)
        # Update the solution array
        sols[i] = z


if __name__ == "__main__":


    for i in range(5):
        '''
        p = multiprocessing.Process(target=worker, args=(i,))
        jobs.append(p)
        p.start()
        '''
        pass
