import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns


class MLP(object):
    def __init__(self, name, path):
        # _dat: Numeric data
        # _: TensorFlow representation
        self.name  = str(name)
        self.path  = str(path) + name + '.mlp'
        self.session = tf.Session()
    def build(self, data, iin, iout, layers):
        # Format for input x and output y
        ndat     = data.shape[0] # Number of data points
        xdat     = data[:, iin]  # The input data
        ydat     = data[:, iout] # The desired output data
        nfeatin  = xdat.shape[1] # Number of input features
        nfeatout = ydat.shape[1] # The number of output features
        # TensorFlow variables to feed input x and desired output y
        x = tf.placeholder("float", [None, nfeatin])
        y = tf.placeholder("float", [None, nfeatout])
        # Insert number of input and output features
        layers  = np.hstack((nfeatin, layers, nfeatout))
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
                pass
            else:
                yp = tf.nn.relu(yp)
        # Store the info!
        self.layers   = layers                # Hidden layer dimensions
        self.xdat     = xdat                  # Input data
        self.ydat     = ydat                  # Output data (targets)
        self.x        = x                     # TF for input data
        self.y        = y                     # TF for output data (targets)
        self.yp       = yp                    # TF for output prediction
        self.w        = w
        self.b        = b
        self.nfeatin  = nfeatin
        self.nfeatout = nfeatout
        self.saver    = tf.train.Saver()
        self.scaler   = StandardScaler()
        self.scaler.fit(self.xdat)
    def train(self, learnrate, trainiter, dispr):
        # Retreive the MLP properties
        xdat = self.xdat
        ydat = self.ydat
        x    = self.x
        y    = self.y
        yp   = self.yp
        # Scale data
        xdat = self.scaler.transform(xdat)
        # Minimise this cost
        cost = tf.reduce_mean(tf.square(y-yp))
        # with this optimiser
        opt  = tf.train.AdamOptimizer(learnrate).minimize(cost)
        # Things to keep track of
        costs   = np.empty(trainiter)
        meancosts = np.empty(trainiter)
        # Initialise variables
        self.session.run(tf.global_variables_initializer())
        # Training
        for i in range(trainiter):
            _, c = self.session.run([opt, cost], feed_dict={x: xdat, y: ydat})
            if i%dispr == 0:
                print("Cost " + str(i) + ": " + str(c))
            costs[i] = c
        # Store the results
        self.ypdat     = self.session.run(yp, feed_dict={x: xdat})
        self.costs     = costs
    def predict(self, xdat):
        # Ensure the correct dimension
        xdat = np.asarray(xdat).reshape(-1, self.nfeatin)
        # Scale as we did in training
        xdat = self.scaler.transform(xdat)
        # Predict
        ypdat = self.session.run(self.yp, feed_dict={self.x: xdat})
        return ypdat
    def plot(self):
        ypdat = self.ypdat
        costs = self.costs
        xdat  = self.xdat
        ydat  = self.ydat
        ul    = np.amax(ydat, axis=0)
        ll    = np.amin(ydat, axis=0)
        count = 1
        for iin in range(self.nfeatin):
            for iout in range(self.nfeatout):
                plt.subplot(self.nfeatin, self.nfeatout, count)
                plt.plot(xdat[:,iin], ydat[:,iout], 'k.', alpha=0.1)
                plt.plot(xdat[:,iin], ypdat[:,iout], '.', alpha=0.3)
                plt.ylim(ll[iout], ul[iout])
                count += 1
        plt.tight_layout()
        plt.show()
        plt.plot(self.costs, 'k--')
        plt.show()
    def save(self):
        return self.saver.save(self.session, self.path)
    def load(self, path):
        self.saver.restore(self.session, path)


def main():

    # Data
    path   = __file__[:-5]
    data   = np.load(path + "Data/ML/Mars_Far.npy")
    iin    = [0,1,2,3,4]
    iout   = [5,6]
    hshape = [10,20,15,]

    dp1 = data[34, iin]
    dp2 = data[34:37, iin]

    # MLP
    mlppath = path + "Data/ML/"
    net = MLP("Mars_Far", mlppath)
    net.build(data, iin, iout, hshape)
    net.load(net.path)
    #net.train(1e-2, 100)
    #net.save()

if __name__ == "__main__":
    main()
