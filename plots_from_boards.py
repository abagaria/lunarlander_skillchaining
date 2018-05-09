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
    parser.add_argument('--skillchain', dest='skillchain', action='store_true')
    parser.add_argument('--no-skillchain', dest='skillchain', action='store_false')
    parser.set_defaults(skillchain=True)
    args = parser.parse_args()

    path = args.logdir + '/' + ("sc" if args.skillchain else "dqn") + '/'
    print "Reading TensorBoard files from", path
    all_rewards = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file[:10] == "events.out" and (subdir[-9:] == "GlobalMDP" or not args.skillchain):
                log_file = subdir + '/' + file
                print log_file
                ea = event_accumulator.EventAccumulator(log_file)
                ea.Reload()
                for tag in ea.Tags()['scalars']:
                    if tag == "Episode_Reward":
                        all_rewards.append(np.array([thing.value for thing in ea.Scalars(tag)]))
                    elif tag == "Epsilon":
                        #TODO
                        pass
    np_all_rewards = np.array(all_rewards)

    '''
    ea = event_accumulator.EventAccumulator("/home/matt/boards_ll/dqn/board_2018_05_09_08_25_19/events.out.tfevents.1525868720.gpu1703")
    ea.Reload()
    for tag in ea.Tags()['scalars']:
        # unicode
        print str(tag)
    """
    for scalar in ea.Scalars('Episode_Reward'):
        print scalar.step, scalar.value
    """
    '''

    '''
    pl.plot(x, y, 'k-')
    pl.fill_between(x, y-error, y+error)
    pl.show()
    '''


if __name__ == '__main__':
    main()