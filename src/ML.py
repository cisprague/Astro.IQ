import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import cProfile

class MLP(object):
    def __init__(self, path):
        self.path    = str(path) + '.mlp'
    def build(self, data, iin, iout, layers):
        # Retreive the data
        self.ndat = data.shape[0] # Number of samples
        self.xdat = data[:,iin]   # Input data
        self.ydat = data[:,iout]  # Output (target) data
        # and scale it
        self.scaler = StandardScaler()
        self.scaler.fit(self.xdat)
        self.xdatsc = self.scaler.transform(self.xdat)
        self.nin = len(iin)
        self.nout = len(iout)
        # TensorFlow variables to feed input x and desired output y
        x = tf.placeholder("float", [None, self.nin], name='Input')
        y = tf.placeholder("float", [None, self.nout], name='Output')
        # Insert number of input and output features
        layers  = np.hstack((self.nin, layers, self.nout))
        # Specifications for assembling the MLP
        nlayers = len(layers) # Number of layers
        iin     = 1           # Index of input layer
        iout    = nlayers - 1 # Index of output layer
        # Assemble the hidden layers
        for i in range(nlayers)[1:]:
            nin  = layers[i-1]
            nout = layers[i]
            w    = tf.Variable(tf.random_normal([nin, nout]), name='w'+str(i))
            b    = tf.Variable(tf.random_normal([nout]), name='b'+str(i))
            # Apply weights
            if i == iin:
                yp = tf.add(tf.matmul(x, w), b, name='yp'+str(i))
            else:
                yp = tf.add(tf.matmul(yp, w), b, name='yp'+str(i))
            # Apply activation
            if i == iout:
                pass
            else:
                yp = tf.nn.relu(yp, name='yp'+str(i))
        # Store the info!
        self.layers = layers # Dimensions
        self.x      = x      # TF input data
        self.y      = y      # TF output data (targets)
        self.yp     = yp     # TF output prediction
        self.w      = w      # TF weights
        self.b      = b      # TF biases
    def train(self, learnrate, trainiter, dispr):
        self.learnrate = learnrate
        # Feed TensorFlow
        self.feed_dict = {self.x: self.xdatsc, self.y: self.ydat}
        # Minimise this cost
        self.cost  = tf.reduce_mean(tf.square(self.y-self.yp))
        # with this optimiser
        self.opt   = tf.train.AdamOptimizer(self.learnrate).minimize(self.cost)
        # and keep track of its progress
        self.costs = np.empty(trainiter)
        saver = tf.train.Saver()
        # Training time!
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(trainiter):
                _, c = sess.run([self.opt, self.cost], self.feed_dict)
                if i%dispr == 0:
                    print("Cost " + str(i) + ": " + str(c))
                self.costs[i] = c

            # Store the results
            self.ypdat     = sess.run(self.yp, self.feed_dict)
            save_path = saver.save(sess, self.path)
            print('Model saved to ' + str(save_path))
    def predict(self, xdat, sess, restore=True):
        if restore:
            saver = tf.train.Saver()
            saver.restore(sess, self.path)
        # Ensure the correct dimension
        xdat = np.asarray(xdat).reshape(-1, self.nin)
        # Scale as we did in training
        xdat = self.scaler.transform(xdat)
        # Predict
        ypdat = sess.run(self.yp, feed_dict={self.x: xdat})
        # reformat if vector
        if ypdat.shape[0] == 1:
            return ypdat[0]
        else:
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
    mlppath = path + "Data/ML/Nets/"
    net1 = MLP(mlppath + "Mars_Far")
    net1.build(data, iin, iout, hshape)
    net1.train(1e-2, 20, 1)
    #print net1.predict(dp1)
    #net1.train(1e-2, 20, 1)
    #net1.save()
    #net1.save()


if __name__ == "__main__":
    main()
    pass
