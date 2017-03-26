'''
Astro.IQ - Data
Christopher Iliffe Sprague
christopher.iliffe.sprague@gmail.com
https://cisprague.github.io/Astro.IQ
'''

from numpy import *
set_printoptions(suppress=True)

def Random_Initial_State(model):
    # Returns random initial state within bounds
    sirange = (model.siub - model.silb)
    sistep  = random.random(model.sdim)*sirange
    return model.silb + sistep
def Random_Initial_States(model, mxstep=10., nstates=5000.):
    # NOTE: Need to make a better way to generate random walks within bounds!
    # model: Dynamical model
    # step: Max percentage step of each state boundary size
    # nstates: Number of initial states to generate
    # The model's initial state boundaries
    silb    = model.silb
    siub    = model.siub
    # The size of the boundary space
    sispace = siub - silb
    # Convert the percent input to number
    mxstep  = mxstep*1e-2*sispace
    # Make array
    states = zeros((nstates, model.sdim))
    # First state in between everything
    states[0] = silb + 0.5*sispace
    for i in arange(1, nstates):
        perturb   = random.randn(model.sdim)*mxstep
        # Reverse and decrease the steps of violating elements
        states[i] = states[i-1] + perturb
        badj = states[i] < silb
        states[i, badj] = states[i-1, badj] - 0.005*perturb[badj]
        badj = states[i] > siub
        states[i, badj] = states[i-1, badj] - 0.005*perturb[badj]
    return states

if __name__ == "__main__":
    ''' -------------------------------------------------
    Problem       : SpaceX Dragon 2 Martian Soft Landing
    Dynamics      : 2-Dimensional Variable Mass Point
    Transcription : Hermite Simpson Seperated (HSS)
    Produces      : Database of fuel-optimal trajectories
    ------------------------------------------------- '''

    # Resources
    from Trajectory import Point_Lander_Drag
    from Optimisation import HSS
    from PyGMO import *
    from numpy import *

    # Load initial states
    print("Loading initial states...")
    si_list = load('Data/Point_Lander_Mars_Initial_States.npy')

    # Define the algorithms to use
    algo_local = algorithm.scipy_slsqp(max_iter=5000, acc=1e-10, screen_output=True)
    algo_meta  = algorithm.mbh(algo_local, stop=1, screen_output=True)

    # Load initial guess
    print("Loading initial guess..")
    z = load('Data/HSS_20_Mars.npy')

    # Allott space for solutions
    n_traj = len(si_list)
    sols   = zeros((n_traj, len(z)))

    # For each initial state
    for i in range(n_traj):
        si    = si_list[i]
        print("Trajectory " + str(i))
        print("State: " + str(si))
        # Initialise the model at that state
        model = Point_Lander_Drag(si)
        # Initialise the HSS problem
        prob  = HSS(model, nsegs=20)
        # Create empty population
        pop   = population(prob)
        # Guess the previous solution
        pop.push_back(z)
        # Optimise from that solution
        print("Beginning optimisation...")
        pop   = algo_meta.evolve(pop)
        # Store the new solution
        z     = pop.champion.x
        # If this is the first trajectory, save the sol, just in case
        if i == 0:
            save("Data/HSS_20_Walk_Base", z)
        # Update the solution array
        sols[i] = z

    # Finally save the solutions
    save("Data/HSS_20_Soltions", sols)
