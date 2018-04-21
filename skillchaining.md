# Create an option to reach the goal
 - Termination function: binary goal trigger function
 - Reward function: R plus option completion reward
 - Initiation set: classification problem
 - Learn policy and initiation set

# Skill chains
 - Termination function: initiation set of first option - add to list of target events
   - When agent first enters new target, create a new option
 - Reward function: R plus option completion reward

# Skill trees
 - "More than one option may be created to reach a target event if that event remains on the target event list after the first option is created to reach it."
 - Conditions:
   - don't create new option if target event is triggered from state already in initiation set of an option targeting that event
   - initiation set does not overlap with siblings or parents
   - limit branching factor by removing target event once it has some number of options targeting it

# Other implementation notes
 - Gestation period
 - 

starts taking 1000 steps after moving to exponential decay

https://www2.cs.duke.edu/research/AI/LSPI/

change point detection after solving mdp - state trajectory through time, fit model to it
Bayesian thing to determine when model is different
scott nikum package champ

fixed timelength pick some number of steps and classify after that
easy hack

bottleneck states best in theory
trajectories into density estimator, sample from estimator
find the peaks, non-maximal supression - ask Ben B
