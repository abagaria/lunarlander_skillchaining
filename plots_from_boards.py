# Matt Corsaro
# Brown University CS 2951X Final Project
# Skill chaining for continuous Lunar Lander
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import os
from os import path
import sys
import random
import argparse

def main():
    parser = argparse.ArgumentParser(description = "Generate plots from TensorBoard log files")
    # If true, generates plots for 
    parser.add_argument("--logdir", type=str, default="/home/matt/boards_ll")
    args = parser.parse_args()

    sc_path = args.logdir + '/' + "sc" + '/'
    dqn_path = args.logdir + '/' + "dqn" + '/'
    rewards = [[], []]
    for i, t in enumerate([sc_path, dqn_path]):
        for subdir, dirs, files in os.walk(t):
            for file in files:
                if file[:10] == "events.out" and (subdir[-9:] == "GlobalMDP" or i):
                    log_file = subdir + '/' + file
                    print "Reading", log_file
                    ea = event_accumulator.EventAccumulator(log_file)
                    ea.Reload()
                    for tag in ea.Tags()['scalars']:
                        if tag == "Episode_Reward":
                            rewards[i].append(np.array([run.value for run in ea.Scalars(tag)]))
                        elif tag == "Epsilon":
                            #TODO
                            pass
    sc_rewards = np.array(rewards[0])
    dqn_rewards = np.array(rewards[1])

    # uncorrected sample sd, population vs sample variance
    sc_avg = np.mean(sc_rewards, axis=0)
    sc_std = np.std(sc_rewards, axis=0)
    dqn_avg = np.mean(dqn_rewards, axis=0)
    dqn_std = np.std(dqn_rewards, axis=0)

    print sc_avg[-1], dqn_avg[-1]

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    
    plt.xlim([0, 1000])
    plt.ylim([-400, 400])
    plt.plot(range(dqn_avg.shape[0]), dqn_avg, 'b')
    plt.fill_between(range(dqn_avg.shape[0]), dqn_avg-dqn_std, dqn_avg+dqn_std)
    plt.savefig(args.logdir + '/' + timestamp + '_dqn.png')
    plt.close()
    plt.xlim([0, 1000])
    plt.ylim([-400, 400])
    plt.plot(range(sc_avg.shape[0]), sc_avg, 'g')
    plt.fill_between(range(sc_avg.shape[0]), sc_avg-sc_std, sc_avg+sc_std)
    plt.savefig(args.logdir + '/' + timestamp + '_sc.png')


if __name__ == '__main__':
    main()