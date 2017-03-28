import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class MLP(object):
    def __init__(self):
        pass
    def build(self, data, iin, iout, layers):

        # Format for input x and output y
        ndat    = data.shape[0] # Number of data points
        xdat    = data[:, iin]  # The input data
        ydat    = data[:, iout] # The desired output data
        ndatin  = xdat.shape[1] # Number of input features
        ndatout = ydat.shape[1] # The number of output features

        # TensorFlow variables to feed input x and desired output y
        x       = tf.placeholder("float", [ndat, ndatin])
        y       = tf.placeholder("float", [ndat, ndatout])

        # Insert number of input and output features
        layers  = np.hstack((ndatin, layers, ndatout))

        # Specifications for assembling the MLP
        nlayers = len(layers) # Number of layers
        iin     = 1           # Index of input layer
        iout    = nlayers - 1 # Index of output layer

        for i in range(nlayers)[1:]:
            nin  = layers[i-1]
            nout = layers[i]
            w    = tf.Variable(tf.random_normal([nin, nout]))
            b    = tf.Variable(tf.random_normal([nout]))

            # Apply weights
            if i == iin:
                yp = tf.add(tf.matmul(x, w), b)
            else:
                yp = tf.add(tf.matmul(yp, w), b)

            # Apply activation
            if i == iout:
                yp = tf.nn.sigmoid(yp)
            else:
                yp = tf.nn.relu(yp)

        # Store the info!
        self.layers      = layers
        self.input_data  = xdat
        self.output_data = ydat
        self.input       = x
        self.output      = y
        self.prediction  = yp

        # Like it? Have a copy!
        return self
    def train(self, learnrate, trainiter):

        # Retreive the MLP properties
        xdat = self.input_data
        ydat = self.output_data
        x    = self.input
        y    = self.output
        yp   = self.prediction

        # Scale data
        scli   = StandardScaler()
        sclo   = StandardScaler()
        xdatsc = scli.fit_transform(xdat)
        ydatsc = sclo.fit_transform(ydat)

        # Minimise this cost
        cost = tf.reduce_mean(tf.square(y-yp))
        # with this optimiser
        opt  = tf.train.GradientDescentOptimizer(learnrate).minimize(cost)

        # Things to keep track of
        costs    = []

        # Begin
        with tf.Session() as sess:

            # Initialise variables
            sess.run(tf.global_variables_initializer())

            # Training
            for i in range(trainiter):
                _, c = sess.run([opt, cost], feed_dict={x: xdatsc, y: ydatsc})
                if i%50 == 0:
                    print("Cost " + str(i) + ": " + str(c))
                costs.append(c)

            # Store the predictions
            yp = sess.run(yp, feed_dict={x: xdatsc})
            self.predictions_us = yp
            self.predictions = sclo.inverse_transform(yp)
            self.costs       = costs



def main():

    # Data
    path   = __file__[:-5]
    data   = np.load(path + "Data/ML/Mars_Far.npy")
    iin    = [0,1,2,3,4]
    iout   = [5,6]
    hshape = [10,20,15,]

    # MLP
    net    = MLP().build(data, iin, iout, hshape)
    net.train(1e-2, 100)

if __name__ == "__main__":
    main()
