from collections import defaultdict, namedtuple
from sympy.matrices import Matrix, ImmutableMatrix
import gym
import numpy as np
import math
import random
import pandas as pd
import json
import os
import pickle
import copy
from random import seed
from random import random

# set up env
env = gym.make('Centipede-ram-v0')
seed(150)

Qtablepath = "sarsa_Qtable.json"
Ntablepath = "sarsa_Ntable.json"


# print(env.action_space) = 18
# print(env.observation_space) = 128

# Q table size :actions^states
# Q_table = [[0]*128 for i in range(18)]
Q_table = defaultdict(float)

# frequency table
N_table = defaultdict(int)

try:
    read_data(Q_table, N_table, "./data/sarsa_data/ntable.txt", "./data/sarsa_data/qtable.txt")
except:
    print("No data to read")
# others 
utility = 0 
steps  = 0
gamma= 0.99
epsilon = 0.99
epsilon_decay = 0.999
final_epsilon = 0.01
alpha = 0.6
alpha_decay = 0.8
SavedState = namedtuple('SavedState', ['state'])

prev_s = None
prev_r = 0
prev_a = 0

def read_data(action_value, action_times, value_path, times_path):
    value_file_size = os.stat(value_path).st_size
    if value_file_size != 0:
        with open(value_path, 'rb') as handle:
            action_value = pickle.loads(handle.read())

    times_file_size = os.stat(times_path).st_size
    if times_file_size != 0:
        with open(times_path, 'rb') as handle:
            action_times = pickle.loads(handle.read())

def save_data(action_value, action_times, value_path, times_path):
    with open(value_path, 'wb') as handle:
        pickle.dump(action_value, handle, protocol=2)
    with open(times_path, 'wb') as handle:
        pickle.dump(action_times, handle, protocol=2)

def epsilon_greedy(s, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    # Pick the action with highest Q value.
    qvals = {a: Q_table[s, a] for a in range(env.action_space.n)}
    max_q = max(qvals.values())

    # In case multiple actions have the same maximum Q value.
    actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
    return np.random.choice(actions_with_max_q)

def update_Q_value(prev_state, action, reward, next_state, done):
    # get the q value for next state with its possible acitons ['possible action' due to epsilon-greedy action selection]
    next_action = epsilon_greedy(next_state, epsilon)
    q_value_next_state = Q_table[next_state, next_action]
    # update N table
    N_table[prev_state, action] = N_table[prev_state, action] + 1
    # update Q table
    # We do not include the value of the next state if terminated.
    Q_table[prev_state, action] += alpha * (
        reward + gamma * q_value_next_state * (1 - done)  - Q_table[prev_state, action])

# run game for a number of runs
samples = 1000
eps_drop = (epsilon - final_epsilon) / samples * 2
for sameple in range(samples):
    # initialize the environment
    curr_s = env.reset()
    utility = 0 
    steps  = 0
    curr_r = 0
    done = False
    step = 0

    while not done:        
        prev_a = epsilon_greedy(SavedState(ImmutableMatrix(curr_s)), epsilon)
        #env.render()
        next_state, reward, done, info = env.step(prev_a)
        if reward is not None:
            prev_r = reward
            utility = utility + reward
        
        # if reward != 0.0:
        #  print(reward)

        prev_s = curr_s
        curr_s = next_state

        update_Q_value(SavedState(ImmutableMatrix(prev_s)), prev_a, reward, SavedState(ImmutableMatrix(curr_s)), done)
        
        steps = steps + 1

    
    if(epsilon > final_epsilon):
         epsilon -= eps_drop
         if(epsilon < final_epsilon):
            epsilon = final_epsilon
        
    print("epsilon:")
    print(epsilon)
    print("steps:")
    print(steps)
    print("Utility value: ")
    print(utility)
    df2 = pd.DataFrame([[steps, utility]], columns=["Steps", "Ut1ility"])
    df2.to_csv("sarsa_out.csv", header=None, mode="a")
    save_data(Q_table, N_table, "./data/sarsa_data/ntable.txt", "./data/sarsa_data/qtable.txt")