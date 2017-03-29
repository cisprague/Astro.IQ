import sys
sys.path.append('../../')
import numpy as np
from ML import MLP
import multiprocessing

''' -----------------------------------------
Here we train 4 different neural networks
architectures in parallel.
>> python Train_20xl_Mars_Far.py
----------------------------------------- '''

def main(nlayers):
    # Load the data to be regressed
    data = np.load('../../Data/ML/Mars_Far.npy')
    # Designate the input and output indicies
    iin  = [0,1,2,3,4]
    iout = [5,6]
    # Define the structure of the hidden layers
    layers = [20]*nlayers
    # Save the neural net here
    path = '../../Data/ML/Nets/Mars_Far_20x' + str(nlayers)
    print path
    # Instantiate the neural network
    net = MLP(path)
    # Build the neural network
    net.build(data, iin, iout, layers)
    # Specify the learning rate
    lr = 1e-4
    # Number of training iterations
    tit = 300000
    # How often to display cost
    dispr = 300
    # Train the network
    print('Beginning training for ' + path)
    net.train(lr, tit, dispr)
    print('Training finished for ' + path)
    np.save(path + '_ypdat', net.ypdat)

if __name__ == "__main__":
    nlayers = [1, 2, 3, 4, 5]
    for i in nlayers:
        p = multiprocessing.Process(target=main, args=(i,))
        p.start()
