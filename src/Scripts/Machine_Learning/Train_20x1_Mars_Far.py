import sys
sys.path.append('../../')
import numpy as np
from ML import MLP

def main():
    # Load the data to be regressed
    data = np.load('../../Data/ML/Mars_Far.npy')
    # Designate the input and output indicies
    iin  = [0,1,2,3,4]
    iout = [5,6]
    # Define the structure of the hidden layers
    layers = [20]
    # The name of the neural network
    name = 'Mars_Far_20x1'
    # The directory to save it
    home = '../../Data/ML/'
    # Instantiate the neural network
    net = MLP(name, home)
    # Build the neural network
    net.build(data, iin, iout, layers)
    # Specify the learning rate
    lr = 1e-4
    # Number of training iterations
    tit = 50000
    # How often to display cost
    dispr = 300
    # Train the network
    print('Beginning training...')
    net.train(lr, tit, dispr)
    print('Training finished!')
    net.save()


if __name__ == "__main__":
    main()
