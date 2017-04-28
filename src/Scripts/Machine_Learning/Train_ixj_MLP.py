import sys
sys.path.append('../../')
import numpy as np
from ML import MLP
import multiprocessing
import itertools
from numba import jit
import cProfile

''' -----------------------------------------
Here we train various different neural
network architectures in parallel.
>> python Train_ixj_MLP.py
----------------------------------------- '''

@jit(nogil=True, cache=True)
def main(width, nlayers):
    # Load the data to be regressed
    path     = '../../Data/Mars/'
    data = np.load(path + 'HSS_10_Train.npy')
    # Designate the input and output indicies
    iin      = [0,1,2,3,4]
    iout     = [5,6]
    # Define the structure of the hidden layers
    layers   = [width]*nlayers
    # Save the neural net here
    path    += 'Nets/' + str('Mars_Reg_New') + '_'+ str(width) + 'x' + str(nlayers)
    # Instantiate the neural network
    net = MLP(path)
    # Build the neural network
    net.build(data, iin, iout, layers, 0.8)
    # Specify the learning rate
    lr = 1e-4
    # Number of training iterations
    tit = 200000
    # How often to display cost
    dispr = 50
    # Train the network
    print('Beginning training for ' + path)
    net.train(lr, tit, dispr)
    print('Training finished for ' + path)


if __name__ == "__main__":
    nlayers = [2, 3]
    width   = [20, 50]
    main(50, 4)
