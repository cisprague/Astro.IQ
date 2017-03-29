import sys
sys.path.append('../../')
import numpy as np
from ML import MLP
import multiprocessing
import itertools

''' -----------------------------------------
Here we train various different neural
network architectures in parallel.
>> python Train_ixj_MLP.py
----------------------------------------- '''

def main(width, nlayers, case):
    # Load the data to be regressed
    path     = '../../Data/ML/'
    pathdata = path + str(case)
    data = np.load(pathdata + '.npy')
    # Designate the input and output indicies
    iin      = [0,1,2,3,4]
    iout     = [5,6]
    # Define the structure of the hidden layers
    layers   = [width]*nlayers
    # Save the neural net here
    path    += 'Nets/' + str(case) + '_'+ str(width) + 'x' + str(nlayers)
    # Instantiate the neural network
    net = MLP(path)
    # Build the neural network
    net.build(data, iin, iout, layers)
    # Specify the learning rate
    lr = 1e-4
    # Number of training iterations
    tit = 200000
    # How often to display cost
    dispr = 300
    # Train the network
    print('Beginning training for ' + path)
    net.train(lr, tit, dispr)
    print('Training finished for ' + path)
    np.save(path + '_ypdat', net.ypdat)

if __name__ == "__main__":
    nlayers = [1, 2, 3]
    width   = [10, 20]
    cases   = ['Mars_Far', 'Mars_Close', 'Mars_Combined']

    for args in itertools.product(width, nlayers, cases):
        p = multiprocessing.Process(target=main, args=args)
        p.start()
