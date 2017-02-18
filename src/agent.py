exptermstepsummaryexpmstoreper'''
Machine Learning Objects
Christopher Iliffe Sprague
christopher.iliffe.sprague@gmail.com
'''

import tensorflow as tf
import numpy      as np
import gym
from collections import defaultdict, collections
import matplotlib.pyplot as plt
import random
import os
import pickle
import time

class TabQAgent(object):
    def __init__(self, env=gym.make('FrozenLake-v0')):
        self.env = env
        self.Q = np.zeros([
            self.env.observation_space.n,
            self.env.action_space.n
        ])
        self.lr =.85
        self.y = .99
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
                if d is True: break
            self.rl.append(RA)

class DQAD(object):
    def __init__(
        self,odim,nact,Qnet,opt,sess,eps=0.05,
        expper=1000,storeper=5,trainper=5,minbatsz=32,
        gamma=0.95,expm=30000,netuprate=0.01,summary=None
        ):
        # memorize arguments
        self.odim       = odim
        self.nact       = nact
        self.q_network  = Qnet
        self.opt        = opt
        self.s          = sess
        self.eps        = eps
        self.expper     = expper
        self.storeper   = storeper
        self.trainper   = trainper
        self.minbatsz   = minbatsz
        self.gamma      = tf.constant(gamma)
        self.expm       = expm
        self.netuprate  = tf.constant(netuprate)
        # deepq state
        self.actiter    = 0
        self.experience = deque()
        self.iteration  = 0
        self.summary    = summary

        self.storeiter  = 0
        self.trainiter  = 0

        self.varinit()

        self.s.run(tf.initialize_all_variables())
        self.s.run(self.target_network_update)

        self.saver = tf.train.Saver()

    def linanneal(self, n, total, p_initial, p_final):
        """Linear annealing between p_initial and p_final
        over total steps - computes value at step n"""
        if n >= total:
            return p_final
        else:
            return p_initial - (n * (p_initial - p_final)) / (total)


    def obsbatchdim(self, batch_size):
        return tuple([batch_size] + list(self.odim))

    def varinit(self):
        self.target_q_network    = self.q_network.copy(scope="target_network")

        # FOR REGULAR ACTION SCORE COMPUTATION
        with tf.name_scope("taking_action"):
            self.observation        = tf.placeholder(tf.float32, self.obsbatchdim(None), name="observation")
            self.action_scores      = tf.identity(self.q_network(self.observation), name="action_scores")
            tf.histogram_summary("action_scores", self.action_scores)
            self.predicted_actions  = tf.argmax(self.action_scores, dimension=1, name="predicted_actions")

        with tf.name_scope("estimating_future_rewards"):
            # FOR PREDICTING TARGET FUTURE REWARDS
            self.next_observation          = tf.placeholder(tf.float32, self.obsbatchdim(None), name="next_observation")
            self.next_observation_mask     = tf.placeholder(tf.float32, (None,), name="next_observation_mask")
            self.next_action_scores        = tf.stop_gradient(self.target_q_network(self.next_observation))
            tf.histogram_summary("target_action_scores", self.next_action_scores)
            self.rewards                   = tf.placeholder(tf.float32, (None,), name="rewards")
            target_values                  = tf.reduce_max(self.next_action_scores, reduction_indices=[1,]) * self.next_observation_mask
            self.future_rewards            = self.rewards + self.gamma * target_values

        with tf.name_scope("q_value_precition"):
            # FOR PREDICTION ERROR
            self.action_mask                = tf.placeholder(tf.float32, (None, self.nact), name="action_mask")
            self.masked_action_scores       = tf.reduce_sum(self.action_scores * self.action_mask, reduction_indices=[1,])
            temp_diff                       = self.masked_action_scores - self.future_rewards
            self.prediction_error           = tf.reduce_mean(tf.square(temp_diff))
            gradients                       = self.opt.compute_gradients(self.prediction_error)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, 5), var)
            # Add histograms for gradients.
            for grad, var in gradients:
                tf.histogram_summary(var.name, var)
                if grad is not None:
                    tf.histogram_summary(var.name + '/gradients', grad)
            self.train_op                   = self.opt.apply_gradients(gradients)

        # UPDATE TARGET NETWORK
        with tf.name_scope("target_network_update"):
            self.target_network_update = []
            for v_source, v_target in zip(self.q_network.variables(), self.target_q_network.variables()):
                # this is equivalent to target = (1-alpha) * target + alpha * source
                update_op = v_target.assign_sub(self.netuprate * (v_target - v_source))
                self.target_network_update.append(update_op)
            self.target_network_update = tf.group(*self.target_network_update)

        # summaries
        tf.scalar_summary("prediction_error", self.prediction_error)

        self.summarize = tf.merge_all_summaries()
        self.no_op1    = tf.no_op()


    def action(self, observation):
        """Given observation returns the action that should be chosen using
        DeepQ learning strategy. Does not backprop."""
        assert observation.shape == self.odim, \
                "Action is performed based on single observation."

        self.actiter += 1
        exploration_p = self.linanneal(self.actiter,
                                              self.expper,
                                              1.0,
                                              self.eps)

        if random.random() < exploration_p:
            return random.randint(0, self.nact - 1)
        else:
            return self.s.run(self.predicted_actions, {self.observation: observation[np.newaxis,:]})[0]

    def expterm(self):
        return min(float(self.actiter) / self.expper, 1.0)

    def store(self, observation, action, reward, newobservation):
        if self.storeiter % self.storeper == 0:
            self.experience.append((observation, action, reward, newobservation))
            if len(self.experience) > self.expm:
                self.experience.popleft()
        self.storeiter += 1

    def step(self):
        """Pick a self.minbatsz exeperiences from reply buffer
        and backpropage the value function.
        """
        if self.trainiter % self.trainper == 0:
            if len(self.experience) <  self.minbatsz:
                return

            # sample experience.
            samples   = random.sample(range(len(self.experience)), self.minbatsz)
            samples   = [self.experience[i] for i in samples]

            # bach states
            states         = np.empty(self.obsbatchdim(len(samples)))
            newstates      = np.empty(self.obsbatchdim(len(samples)))
            action_mask    = np.zeros((len(samples), self.nact))

            newstates_mask = np.empty((len(samples),))
            rewards        = np.empty((len(samples),))

            for i, (state, action, reward, newstate) in enumerate(samples):
                states[i] = state
                action_mask[i] = 0
                action_mask[i][action] = 1
                rewards[i] = reward
                if newstate is not None:
                    newstates[i] = newstate
                    newstates_mask[i] = 1
                else:
                    newstates[i] = 0
                    newstates_mask[i] = 0


            calculate_summaries = self.iteration % 100 == 0 and \
                    self.summary is not None

            cost, _, summary_str = self.s.run([
                self.prediction_error,
                self.train_op,
                self.summarize if calculate_summaries else self.no_op1,
            ], {
                self.observation:            states,
                self.next_observation:       newstates,
                self.next_observation_mask:  newstates_mask,
                self.action_mask:            action_mask,
                self.rewards:                rewards,
            })

            self.s.run(self.target_network_update)

            if calculate_summaries:
                self.summary.add_summary(summary_str, self.iteration)

            self.iteration += 1

        self.trainiter += 1

if __name__ == '__main__':
    TA = TabQAgent()
    TA.Learn()
