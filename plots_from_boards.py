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

def savePlot(rewards, color, filename):
    plt.xlim([0, 1000])
    plt.ylim([-400, 400])
    avg = np.mean(rewards, axis=0)
    smooth_average = smooth(avg, 0.9)
    std = np.std(rewards, axis=0)
    plt.plot(range(avg.shape[0]), avg, 'b')
    plt.plot(range(smooth_average.shape[0]), smooth_average, 'k')
    plt.fill_between(range(avg.shape[0]), avg-std, avg+std)
    plt.savefig(filename)
    plt.close()
    return (avg, smooth_average)

def plotAll(averages, filename):
    plt.xlim([0, 1000])
    plt.ylim([-400, 400])
    colors = [['b', 'aqua'], ["purple", "fuchsia"], ['r', 'm'], ['g', 'y']]
    for i, (avg, sm) in enumerate(averages):
        plt.plot(range(avg.shape[0]), avg, colors[i][1])
    for i, (avg, sm) in enumerate(averages):
        plt.plot(range(sm.shape[0]), sm, colors[i][0])
    plt.savefig(filename)
    plt.close()

# https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar
# Replicate tensorboard smoothed plotting function
def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return np.array(smoothed)

def main():
    parser = argparse.ArgumentParser(description = "Generate plots from TensorBoard log files")
    # If true, generates plots for 
    parser.add_argument("--logdir", type=str, default="/home/matt/boards_ll")
    args = parser.parse_args()

    paths = []
    paths.append(args.logdir + '/' + "dqn" + '/')
    paths.append(args.logdir + '/' + "sc_load_dqn" + '/')
    paths.append(args.logdir + '/' + "sc" + '/')
    paths.append(args.logdir + '/' + "sc_epsilon_cutoff" + '/')
    rewards = [[] for p in range(len(paths))]
    for i, t in enumerate(paths):
        for subdir, dirs, files in os.walk(t):
            for file in files:
                if file[:10] == "events.out" and (subdir[-9:] == "GlobalMDP" or not i):
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
    dqn_rewards = np.array(rewards[0])
    scldqn_rewards = np.array(rewards[1])
    sc_rewards = np.array(rewards[2])
    scec_rewards = np.array(rewards[3])

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')

    averages = []
    averages.append(savePlot(dqn_rewards, 2, args.logdir + '/' + timestamp + '_dqn.png'))
    averages.append(savePlot(scldqn_rewards, 2, args.logdir + '/' + timestamp + '_scldqn.png'))
    averages.append(savePlot(sc_rewards, 2, args.logdir + '/' + timestamp + '_sc.png'))
    averages.append(savePlot(scec_rewards, 2, args.logdir + '/' + timestamp + '_scec.png'))

    plotAll(averages, args.logdir + '/' + timestamp + '_all.png')
    plotAll(averages[:2], args.logdir + '/' + timestamp + '_scl_and_dqn.png')

if __name__ == '__main__':
    main()