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
import matplotlib.pyplot as plt

import time
import datetime
import os
from os import path
import sys
import random
from collections import deque
from anytree import NodeMixin, RenderTree

import argparse

# TODO: This code's pretty messy..

# TODO: Capitalize, or don't make them global
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
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_length = 10000
epsilon_decay_exp = 0.98

# Skill chain params
# don't execute after creating, off-policy learning
gestation = 10
# Stop adding options after this timestep
add_opt_cutoff = num_episodes/5
# Maximum number of steps in one option
max_steps_opt = 25
max_neg_traj = max_steps_opt*10
# Option completion reward - not used since global MDP currently must choose an option if presented with it
opt_r = 35
# How long to gather initiation classifier data for, and the maximum number of examples that can be reached before
num_ep_init_class = 50
max_num_init_ex = 6000
# unused
max_branching_factor = 2
# episode to drop the epsilon to 0
epsilon_drop_episode = 4*num_episodes/5

def atGoal(state, done):
    # If landed in the target zone (between the two flags)
    x = state[0]
    y = state[1]
    return -0.2 < x < 0.2 and -0.1 < y < 0.1 and done

def getMinibatchElem(minibatch, i):
    return np.asarray([elem[i] for elem in minibatch])

def statesFromExperiences(experiences):
    return [example[0][:2] for example in experiences]

def make_meshgrid(x_min, x_max, y_min, y_max, h=.02):
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# TODO: BFS function
def findOptForState(position, root_option, ep):
    # BFS Search
    queue = [root_option]
    while len(queue) != 0:
        opt = queue.pop(0)
        # If the state is in the initation set and the initiation set classifier has been fully trained
        if opt.inInitiationSet(position) and opt.classifierTrained():
            return opt
        else:
            queue += opt.children
    return None

def writeAllEpsilon(root_option, ep):
    # BFS iteration
    queue = [root_option]
    while len(queue) != 0:
        option = queue.pop(0)
        option.writeEpsilon(ep)
        queue += option.children

def dropAllEpsilon(root_option):
    # BFS iteration
    queue = [root_option]
    while len(queue) != 0:
        option = queue.pop(0)
        option.epsilon = 0.0
        queue += option.children

def main():

    parser = argparse.ArgumentParser(description = "Lunar Lander")
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.add_argument('--no-visualize', dest='visualize', action='store_false')
    parser.set_defaults(visualize=False)
    args = parser.parse_args()

    # game parameters
    env = gym.make("LunarLander-v2")
    state_dim = np.prod(np.array(env.observation_space.shape))
    n_actions = env.action_space.n

    # set seeds to 0
    #env.seed(0)
    #np.random.seed(0)

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

    plot_epsilon = tf.Variable(0.)
    tf.summary.scalar("Epsilon", plot_epsilon)
    eps_summary_placeholder = tf.placeholder("float")
    update_plot_epsilon = plot_epsilon.assign(eps_summary_placeholder)

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

    # date and time, with full unix timestamp appended
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S___') + str(int(time.time() *10e5))
    class Option:
        def __init__(self, n, start_ep):
            self.n = n
            self.start_ep = start_ep

            self.sess = tf.Session()
            # TODO: Initialize from globalMDP - all the way up the tree - possibly use saver and loader 
            self.sess.run(tf.global_variables_initializer())

            self.writer = tf.summary.FileWriter("board_" + timestamp + '_' + str(n))
            self.writer.add_graph(self.sess.graph)

            self.saver = tf.train.Saver()

            self.directory = timestamp + '/' + str(self.n)
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)

            self.initiation_examples = []
            self.initiation_labels = []
            self.initiation_classifier = svm.SVC(kernel="rbf")

            self.experience = deque(maxlen=replay_memory_capacity)

            self.gestation = True
            self.initTrained = False

            self.epsilon = epsilon_start
            self.epsilon_linear_step = (epsilon_start-epsilon_end)/epsilon_decay_length
            self.total_steps = 0

        def writeReward(self, r, ep):
            self.sess.run(update_ep_reward, feed_dict={r_summary_placeholder: r})
            summary_str = self.sess.run(tf.summary.merge_all())
            self.writer.add_summary(summary_str, ep)

        def writeEpsilon(self, ep):
            self.sess.run(update_plot_epsilon, feed_dict={eps_summary_placeholder: self.epsilon})
            if self.n != "GlobalMDP":
                summary_str = self.sess.run(tf.summary.merge_all())
                self.writer.add_summary(summary_str, ep)

        def retrainInitationClassifier(self, ep):
            self.num_pos_examples = len([x for x in self.initiation_labels if x == 1])
            self.num_neg_examples = len([x for x in self.initiation_labels if x == 0])
            if self.num_pos_examples != 0 and self.num_neg_examples != 0:
                print "Training classifier with", len([x for x in self.initiation_labels if x == 1]), \
                    "positive examples and", len([x for x in self.initiation_labels if x == 0]), "negative examples." 
                class_start_time = time.time()
                self.initiation_classifier.fit(self.initiation_examples, self.initiation_labels)
                print "Retrained option", self.n, "classifier in", (time.time() - class_start_time), "seconds."
                self.saveInitiationPlot(ep)
                self.initTrained = True
            #else:
                #print "Not training classifier,", len([x for x in self.initiation_labels if x == 1]), \
                #    "positive examples and", len([x for x in self.initiation_labels if x == 0]), "negative examples."

        def classifierTrained(self):
            return ep - self.start_ep > num_ep_init_class or len(self.initiation_labels) > max_num_init_ex        

        def loadDQNWeights(self, model_file):
            print "Loading weights for new option", self.n, "from", model_file
            self.saver.restore(self.sess, model_file)

        def saveDQNWeights(self, model_file):
            assert(self.n == "GlobalMDP")
            print "Saving", self.n, "DQN weights to", model_file
            self.saver.save(self.sess, model_file)

        def saveInitiationPlot(self, ep):
            try:
                # very rarely, the legend doesn't fit correctly, and this fails
                # http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
                X0, X1 = np.array(self.initiation_examples)[:, 0], np.array(self.initiation_examples)[:, 1]
                xx, yy = make_meshgrid(-1, 1, -1./3, 1)
                labels = [str(self.num_pos_examples) + " positive examples", \
                    str(self.num_neg_examples) + " negative examples"]
                
                fig, sub = plt.subplots(1, 1)

                plot_contours(sub, self.initiation_classifier, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
                sub.scatter(X0, X1, c=self.initiation_labels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
                sub.set_xlim(xx.min(), xx.max())
                sub.set_ylim(yy.min(), yy.max())
                sub.set_xticks(())
                sub.set_yticks(())
                sub.set_xlabel("Option " + str(self.n) + " at episode " + str(ep))
                sub.set_ylabel(str(self.num_pos_examples) + " pos, " + str(self.num_neg_examples) + " neg")

                #sub.legend(labels=labels, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
                plt.plot([-0.2, 0.2], [0, 0], 'k-')
                plt.savefig(self.directory + '/' + str(ep) + '.png')

                plt.close()
            except:
                print "Failed to generate plot for option", self.n, " at episode", ep
                print sys.exc_info()[0]

        def addInitiationExample(self, state, label):
            self.initiation_examples.append(state)
            self.initiation_labels.append(label)

        def addInitiationExamples(self, states, label):
            
            self.initiation_examples += states
            self.initiation_labels += [label]*len(states)

        def inInitiationSet(self, state):
            return self.initTrained and self.initiation_classifier.predict([state])[0]

        # TODO: epsilon decay
        def updateEpsilon(self, done, ep):
            # Only update if Epsilon hasn't been forced to zero
            if ep < epsilon_drop_episode:
                # linearly decay epsilon from epsilon_start to epsilon_end over epsilon_decay_length steps
                decay = ""
                old_epsilon = self.epsilon
                if self.total_steps < epsilon_decay_length:
                    self.epsilon -= self.epsilon_linear_step
                    decay = "linear"
                # then exponentially decay it every episode
                elif done:
                    self.epsilon *= epsilon_decay_exp
                    decay = "exponential"
                #print "Updating option", self.n, "epsilon from", old_epsilon, "to", self.epsilon, "with", decay, "decay."

        def updateDQN(self, step_experience):
            self.experience.append(step_experience)

            # update the slow target's weights to match the latest q network if it's time to do so
            if self.total_steps%update_slow_target_every == 0:
                _ = self.sess.run(update_slow_target_op)

            # update network weights to fit a minibatch of experience
            if self.total_steps%train_every == 0 and len(self.experience) >= minibatch_size:

                # grab N (s,a,r,s') tuples from experience
                minibatch = random.sample(self.experience, minibatch_size)

                # do a train_op with all the inputs required

                _ = self.sess.run(train_op,
                    feed_dict = {state_ph: getMinibatchElem(minibatch, 0), action_ph: getMinibatchElem(minibatch, 1), \
                        reward_ph: getMinibatchElem(minibatch, 2), next_state_ph: getMinibatchElem(minibatch, 3), \
                        is_not_terminal_ph: getMinibatchElem(minibatch, 4), is_training_ph: True})

    # http://anytree.readthedocs.io/en/latest/api/anytree.node.html#anytree.node.nodemixin.NodeMixin
    class Skill(Option, NodeMixin):
        def __init__(self, n, start_ep, parent = None):
            super(Skill, self).__init__(n, start_ep)
            self.parent = parent
            self.name = str(n)

            # Not the global MDP or the goal option
            if self.parent != None and self.parent.parent != None:
                global_mdp = self.parent
                while global_mdp.parent != None:
                    global_mdp = global_mdp.parent
                # Timestamp shouldn't be necessary since only one model will be saved for each option
                timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
                weights_file = self.directory + '/' + timestamp + ".ckpt"
                global_mdp.saveDQNWeights(weights_file)
                self.loadDQNWeights(weights_file)


        def inTerminationSet(self, full_state, done):
            if self.n == "GlobalMDP":
                return False
            elif self.n == 0:
                return atGoal(full_state, done)
            else:
                return self.parent.inInitiationSet(full_state[:2])

        def updateInit(self, experiences, ep):
            # Only called if `self.inTerminationSet(experiences[-1][0], (not experiences[-1][-1]))`
            if not self.classifierTrained():
                # List of (x, y) states for experiences less than max_steps_opt time steps away from the goal
                positive_examples = statesFromExperiences(experiences[-max_steps_opt:])
                # Only use the last max_neg_traj negative examples, not the hovering at the beginning
                negative_examples = statesFromExperiences(experiences[-max_steps_opt-max_neg_traj:-max_steps_opt])
                # If there aren't examples or if first negative example isn't already in the initiation set
                if len(self.initiation_examples) == 0 or len(negative_examples) == 0 or not self.inInitiationSet(negative_examples[0]):
                    self.addInitiationExamples(positive_examples, 1)
                    self.addInitiationExamples(negative_examples, 0)
                    self.retrainInitationClassifier(ep)
                elif len(negative_examples) != 0:
                    print "Trajectory began at state", negative_examples[0], "which is in the initiation set. Skipping."

    # initialize session
    globalMDP = Skill("GlobalMDP", 0)

    num_skills = 0
    goalOpt = Skill(num_skills, 0, parent=globalMDP)
    num_skills += 1
    new_opt = goalOpt

    #####################################################################################################
    ## Training

    start_time = time.time()
    for ep in range(num_episodes):
        newopt_episode_terminated = False

        total_reward = 0
        raw_reward = 0
        steps_in_ep = 0

        epi_experience = []

        observation = env.reset()

        opt = globalMDP
        if new_opt != None and new_opt.classifierTrained():
            new_opt = None
        if ep >= epsilon_drop_episode:
            dropAllEpsilon(globalMDP)
        for t in range(max_steps_ep):
            current_position = observation[:2]

            if opt == globalMDP:
                current_opt = findOptForState(current_position, goalOpt, ep)
                if current_opt != None:
                    opt = current_opt
                    print "Switching from global MDP to option", opt.name
                    # When transitioning to option from global, and no option is being initialized
                    if new_opt == None and ep < add_opt_cutoff:
                        print "Creating a new option with parent", opt.name
                        new_opt = Skill(num_skills, ep, parent=opt)
                        num_skills += 1
                else:
                    opt = globalMDP
            
            if np.random.random() < opt.epsilon:
                action = np.random.randint(n_actions)
            else:
                q_s = opt.sess.run(q_action_values, feed_dict = {state_ph: observation[None], is_training_ph: False})
                action = np.argmax(q_s)

            # take step
            next_observation, reward, done, _info = env.step(action)
            if args.visualize:
                env.render()

            opt_reward = reward

            # Since we aren't allowing the global MDP to choose between an action and an option, don't need to give
            # extra completion reward to encourage choosing an option. Therefore, no difference between total and raw
            # reward
            '''
            # if current option is completed and we move to the next, or if we've reached the goal with the goal option
            if (opt == goalOpt and done) or \
                (opt != globalMDP and opt != goalOpt and opt.parent.inInitiationSet(next_observation[0][:2])):
                print "Completed opt", opt.name, " Moving to opt", opt.parent.name
                opt_reward += opt_r
                opt = opt.parent
            '''

            total_reward += opt_reward
            raw_reward += reward

            step_experience = (observation, action, opt_reward, next_observation, 0.0 if done else 1.0)

            opt.updateDQN(step_experience)
            epi_experience.append(step_experience)

            observation = next_observation
            opt.total_steps += 1
            steps_in_ep += 1

            opt.updateEpsilon(done, ep)
            if opt != globalMDP:
                globalMDP.updateDQN(step_experience)
                globalMDP.updateEpsilon(done, ep)
                if opt.inTerminationSet(observation, done):
                    print "Switching from option", opt.name, "to option", opt.parent.name
                    opt = opt.parent
            
            if new_opt != None and not new_opt.classifierTrained() and new_opt.inTerminationSet(observation, done) and not newopt_episode_terminated:
                # Only update once per episode, at what would be the transition
                newopt_episode_terminated = True
                for exp in epi_experience[-max_steps_opt:]:
                    new_opt.updateDQN(exp)
                new_opt.updateInit(epi_experience, ep)

            if done:
                # Increment episode counter
                _ = opt.sess.run(episode_inc_op)
                break
        
        # TODO: only write once, writeEpsilon currently writes for all but global since nothing else is plotted
        writeAllEpsilon(globalMDP, ep)
        globalMDP.writeReward(raw_reward, ep)

        print('Episode %2i, Reward: %7.3f, Steps: %i, Minutes: %7.3f'%\
            (ep, raw_reward, steps_in_ep, (time.time() - start_time)/60))

    env.close()

if __name__ == '__main__':
    main()
