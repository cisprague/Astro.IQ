'''
Machine Learning Objects
Christopher Iliffe Sprague
christopher.iliffe.sprague@gmail.com
'''

import tensorflow as tf
import numpy      as np
import gym
from collections import defaultdict
import matplotlib.pyplot as plt

class DQNAgent(object):

    def __init__(self, env):
        self.env     = env    # An agent has no purpose without an environemnt
        self.gamma   = 0.9    # Decay rate (diminished future reward)
        self.eps     = 0.7    # Probability of random action (exploration)
        self.epsdec  = 0.99   # Explore less as we learn
        self.epsmin  = 0.05   # Minimum exploration probability
        self.lrnrate = 1e-4   # Amount learnt per iteration
        return None

class TabQAgent(object):

    def __init__(self, env=gym.make('FrozenLake-v0')):
        self.env = env
        self.Q = np.zeros([
            self.env.observation_space.n,
            self.env.action_space.n
        ])
        self.lr =.85
        self.y = .99
        self.jl = []
        self.rl = []

    def Learn(self,neps=200,nits=100):
        for e in range(neps):
            s = self.env.reset()
            RA = 0
            d = False
            for i in range(nits):
                a = np.argmax(self.Q[s,:] + np.random.randn(1,self.env.action_space.n)*(1./(e+1)))
                s1,r,d,_ = self.env.step(a)
                self.Q[s,a] = self.Q[s,a] + self.lr*(r + self.y*np.max(self.Q[s1,:]) - self.Q[s,a])
                RA += r
                s = s1
                if d is True:
                    break
            self.rl.append(RA)
        plt.plot(self.rl)
        plt.show()

if __name__ == '__main__':
    TA = TabQAgent()
    TA.Learn()
