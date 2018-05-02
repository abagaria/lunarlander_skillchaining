# Matt Corsaro
# Brown University CS 2951X Final Project
# Skill chaining for continuous Lunar Lander
# Original DQN code from:
# https://gist.github.com/heerad/d2b92c2f3a83b5e4be395546c17b274c#file-dqn-lunarlander-v2-py
import numpy as np
import gym
from gym import wrappers
import tensorflow as tf
from sklearn import svm

import time
import datetime
import os
from os import path
import sys
import random
from collections import deque
from anytree import NodeMixin, RenderTree

import argparse

def getMinibatchElem(minibatch, i):
    return np.asarray([elem[i] for elem in minibatch])

def statesFromExperiences(experiences):
    return [example[0][:2] for example in experiences]

def main():

    parser = argparse.ArgumentParser(description = "Lunar Lander")
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.add_argument('--no-visualize', dest='visualize', action='store_false')
    parser.set_defaults(feature=False)
    args = parser.parse_args()

    # DQN Params
    gamma = 0.99
    # Hidden layer sizes
    h1 = 200
    h2 = 200
    h3 = 200
    lr = 5e-5
    # decay per episode
    lr_decay = 1
    l2_reg = 1e-6
    dropout = 0
    num_episodes = 1000
    # gym cuts off after 1000, anyway
    max_steps_ep = 1000
    update_slow_target_every = 100
    train_every = 1
    replay_memory_capacity = int(1e6)
    minibatch_size = 1024
    #TODO: Get epsilon close to zero by ep 50
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_length = 10000
    epsilon_decay_exp = 0.98

    # Skill chain params
    # don't execute after creating, off-policy learning
    gestation = 10
    # Stop adding options after this timestep
    add_opt_cutoff = num_episodes/2
    # Maximum number of steps in one option
    max_steps_opt = max_steps_ep/10
    # Option completion reward
    opt_r = 35

    # How long to wait before adding new option?
    steps_per_opt = num_episodes/10

    # game parameters
    env = gym.make("LunarLander-v2")
    state_dim = np.prod(np.array(env.observation_space.shape))
    n_actions = env.action_space.n

    # set seeds to 0
    env.seed(0)
    np.random.seed(0)

    ####################################################################################################################
    ## Tensorflow

    tf.reset_default_graph()

    # placeholders
    state_ph = tf.placeholder(dtype=tf.float32, shape=[None,state_dim]) # input to Q network
    next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None,state_dim]) # input to slow target network
    action_ph = tf.placeholder(dtype=tf.int32, shape=[None]) # action indices (indices of Q network output)
    reward_ph = tf.placeholder(dtype=tf.float32, shape=[None]) # rewards (go into target computation)
    is_not_terminal_ph = tf.placeholder(dtype=tf.float32, shape=[None]) # indicators (go into target computation)
    is_training_ph = tf.placeholder(dtype=tf.bool, shape=()) # for dropout

    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Episode Reward", episode_reward)
    r_summary_placeholder = tf.placeholder("float")
    update_ep_reward = episode_reward.assign(r_summary_placeholder)

    # episode counter
    episodes = tf.Variable(0.0, trainable=False, name='episodes')
    episode_inc_op = episodes.assign_add(1)

    # will use this to initialize both Q network and slowly-changing target network with same structure
    def generate_network(s, trainable, reuse):
        hidden = tf.layers.dense(s, h1, activation = tf.nn.relu, trainable = trainable, name = 'dense', reuse = reuse)
        hidden_drop = tf.layers.dropout(hidden, rate = dropout, training = trainable & is_training_ph)
        hidden_2 = tf.layers.dense(hidden_drop, h2, activation = tf.nn.relu, trainable = trainable, name = 'dense_1', \
            reuse = reuse)
        hidden_drop_2 = tf.layers.dropout(hidden_2, rate = dropout, training = trainable & is_training_ph)
        hidden_3 = tf.layers.dense(hidden_drop_2, h3, activation = tf.nn.relu, trainable = trainable, name = 'dense_2',\
            reuse = reuse)
        hidden_drop_3 = tf.layers.dropout(hidden_3, rate = dropout, training = trainable & is_training_ph)
        action_values = tf.squeeze(tf.layers.dense(hidden_drop_3, n_actions, trainable = trainable, name = 'dense_3', \
            reuse = reuse))
        return action_values

    with tf.variable_scope('q_network') as scope:
        # Q network applied to state_ph
        q_action_values = generate_network(state_ph, trainable = True, reuse = False)
        # Q network applied to next_state_ph (for double Q learning)
        q_action_values_next = tf.stop_gradient(generate_network(next_state_ph, trainable = False, reuse = True))

    # slow target network
    with tf.variable_scope('slow_target_network', reuse=False):
        # use stop_gradient to treat the output values as constant targets when doing backprop
        slow_target_action_values = tf.stop_gradient(generate_network(next_state_ph, trainable = False, reuse = False))

    # isolate vars for each network
    q_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_network')
    slow_target_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_network')

    # update values for slowly-changing target network to match current critic network
    update_slow_target_ops = []
    for i, slow_target_var in enumerate(slow_target_network_vars):
        update_slow_target_op = slow_target_var.assign(q_network_vars[i])
        update_slow_target_ops.append(update_slow_target_op)

    update_slow_target_op = tf.group(*update_slow_target_ops, name='update_slow_target')

    targets = reward_ph + is_not_terminal_ph * gamma * \
        tf.gather_nd(slow_target_action_values, tf.stack((tf.range(minibatch_size), \
            tf.cast(tf.argmax(q_action_values_next, axis=1), tf.int32)), axis=1))

    # Estimated Q values for (s,a) from experience replay
    estim_taken_action_vales = tf.gather_nd(q_action_values, tf.stack((tf.range(minibatch_size), action_ph), axis=1))

    # loss function (with regularization)
    loss = tf.reduce_mean(tf.square(targets - estim_taken_action_vales))
    for var in q_network_vars:
        if not 'bias' in var.name:
            loss += l2_reg * 0.5 * tf.nn.l2_loss(var)

    # optimizer
    train_op = tf.train.AdamOptimizer(lr*lr_decay**episodes).minimize(loss)

    ## Tensorflow
    ####################################################################################################################

    board_name = datetime.datetime.fromtimestamp(time.time()).strftime('board_%Y_%m_%d_%H_%M_%S')
    class Option:
        def __init__(self, n):
            self.n = n

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

            self.writer = tf.summary.FileWriter(board_name + '_' + str(n))
            self.writer.add_graph(self.sess.graph)

            self.initiation_examples = []
            self.initiation_labels = []
            self.initiation_classifier = svm.SVC(kernel="rbf")

            self.experience = deque(maxlen=replay_memory_capacity)

            self.gestation = True

            self.epsilon = epsilon_start
            self.epsilon_linear_step = (epsilon_start-epsilon_end)/epsilon_decay_length
            self.total_steps = 0

        def writeReward(self, r, ep):
            self.sess.run(update_ep_reward, feed_dict={r_summary_placeholder: r})
            summary_str = self.sess.run(tf.summary.merge_all())
            self.writer.add_summary(summary_str, ep)

        def retrainInitationClassifier(self):
            if len([x for x in self.initiation_labels if x == 1]) != 0 and \
                len([x for x in self.initiation_labels if x == 0]) != 0:
                print "Training classifier with", len([x for x in self.initiation_labels if x == 1]), \
                    "positive examples and", len([x for x in self.initiation_labels if x == 0]), "negative examples." 
                class_start_time = time.time()
                self.initiation_classifier.fit(self.initiation_examples, self.initiation_labels)
                print "Retrained option", self.n, "classifier in", (time.time() - class_start_time), "seconds."
            else:
                print "Not training classifier,", len([x for x in self.initiation_labels if x == 1]), \
                    "positive examples and", len([x for x in self.initiation_labels if x == 0]), "negative examples."

        def addInitiationExample(self, state, label):
            self.initiation_examples.append(state)
            self.initiation_labels.append(label)

        def addInitiationExamples(self, states, label):
            self.initiation_examples += states
            self.initiation_labels += [label]*len(states)

        def inInitiationSet(self, state):
            return self.initiation_classifier.predict([state])[0]

        def updateEpsilon(self, done):
            # linearly decay epsilon from epsilon_start to epsilon_end over epsilon_decay_length steps
            if self.total_steps < epsilon_decay_length:
                self.epsilon -= self.epsilon_linear_step
            # then exponentially decay it every episode
            elif done:
                self.epsilon *= epsilon_decay_exp

    # http://anytree.readthedocs.io/en/latest/api/anytree.node.html#anytree.node.nodemixin.NodeMixin
    class Skill(Option, NodeMixin):
        def __init__(self, n, parent = None):
            super(Skill, self).__init__(n)
            self.parent = parent
            self.name = str(n)

    # initialize session
    opt = Skill(0)
    target_positions = []

    #####################################################################################################
    ## Training

    start_time = time.time()
    for ep in range(num_episodes):

        total_reward = 0
        raw_reward = 0
        steps_in_ep = 0

        ep_experience = []

        observation = env.reset()

        for t in range(max_steps_ep):

            current_position = observation[:2]

            # TODO: Choose an action, option, or random
            if np.random.random() < opt.epsilon:
                action = np.random.randint(n_actions)
            else:
                q_s = opt.sess.run(q_action_values, feed_dict = {state_ph: observation[None], is_training_ph: False})
                action = np.argmax(q_s)

            # take step
            next_observation, reward, done, _info = env.step(action)
            if args.visualize:
                env.render()

            opt_reward = reward + 0
            total_reward += opt_reward
            raw_reward += reward

            # add this to experience replay buffer
            opt.experience.append((observation, action, opt_reward, next_observation, 0.0 if done else 1.0))
            ep_experience.append((observation, action, opt_reward, next_observation, 0.0 if done else 1.0))

            # update the slow target's weights to match the latest q network if it's time to do so
            if opt.total_steps%update_slow_target_every == 0:
                _ = opt.sess.run(update_slow_target_op)

            # update network weights to fit a minibatch of experience
            if opt.total_steps%train_every == 0 and len(opt.experience) >= minibatch_size:

                # grab N (s,a,r,s') tuples from experience
                minibatch = random.sample(opt.experience, minibatch_size)

                # do a train_op with all the inputs required

                _ = opt.sess.run(train_op,
                    feed_dict = {state_ph: getMinibatchElem(minibatch, 0), action_ph: getMinibatchElem(minibatch, 1), \
                        reward_ph: getMinibatchElem(minibatch, 2), next_state_ph: getMinibatchElem(minibatch, 3), \
                        is_not_terminal_ph: getMinibatchElem(minibatch, 4), is_training_ph: True})
            observation = next_observation
            opt.total_steps += 1
            steps_in_ep += 1

            opt.updateEpsilon(done)

            if done:
                # Increment episode counter
                _ = opt.sess.run(episode_inc_op)
                break

        # If ended in the target zone (between the two flags) and done == True
        if -0.2 < ep_experience[-1][0][0] < 0.2 and not ep_experience[-1][-1]:
            # Last max_steps_opt experiences before reaching the goal
            initiation_experiences = ep_experience[-max_steps_opt:]
            # List of (x, y) states for experiences more than 100 time steps away from the goal
            positive_examples = statesFromExperiences(ep_experience[-max_steps_opt:])
            negative_examples = statesFromExperiences(ep_experience[:-max_steps_opt])
            opt.addInitiationExamples(positive_examples, 1)
            opt.addInitiationExamples(negative_examples, 0)
            opt.retrainInitationClassifier()
        else:
            # The goal wasn't reached, so all episodes aren't part of the initiation set
            # However, negative would then greatly outnumber positive..
            '''
            negative_examples = statesFromExperiences(ep_experience)
            opt.addInitiationExamples(negative_examples, 0)
            opt.retrainInitationClassifier()
            '''

        opt.writeReward(raw_reward, ep)

        print('Episode %2i, Reward: %7.3f, Steps: %i, Next eps: %7.3f, Minutes: %7.3f'%\
            (ep, raw_reward, steps_in_ep, opt.epsilon, (time.time() - start_time)/60))

    env.close()

if __name__ == '__main__':
    main()