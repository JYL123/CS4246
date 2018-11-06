from collections import defaultdict, namedtuple
from sympy.matrices import Matrix, ImmutableMatrix
import gym
import numpy as np
import math
import random

# set up env
env = gym.make('Centipede-ram-v0')

# print(env.action_space) = 18
# print(env.observation_space) = 128

# Q table size :actions^states
# Q_table = [[0]*128 for i in range(18)]
Q_table = defaultdict(float)
# frequency table
N_table = defaultdict(int)
# others 
utility = 0 
steps  = 0
gamma= 0.99
epsilon = 0.1
SavedState = namedtuple('SavedState', ['state'])


prev_s = None
prev_r = 0
prev_a = 0

def epsilon_greedy(s):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    # Pick the action with highest Q value.
    qvals = {a: Q_table[s, a] for a in range(env.action_space.n)}
    max_q = max(qvals.values())

    # In case multiple actions have the same maximum Q value.
    actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
    return np.random.choice(actions_with_max_q)

def update_Q_value(prev_state, action, reward, next_state):
    next_action = epsilon_greedy(next_state)
    q_value_next_state = Q_table[next_state, next_action]
    # update N table
    N_table[prev_state, action] = N_table[prev_state, action] + 1
    # We do not include the value of the next state if terminated.
    Q_table[prev_state, action] += (1/N_table[prev_state, action]) * (
        reward + gamma * q_value_next_state  - Q_table[prev_state, action])

print(Q_table)

# run game for a number of runs
samples = 100
for sameple in range(samples):
    # initialize the environment
    curr_s = env.reset()
    curr_r = 0
    done = False
    step = 0

    while not done:        
        prev_a = epsilon_greedy(SavedState(ImmutableMatrix(curr_s)))
        
        
        next_state, reward, done, info = env.step(prev_a)
        if reward is not None:
            prev_r = reward
            utility = utility + reward
        
        if reward != 0.0:
         print(reward)

        prev_s = curr_s
        curr_s = next_state
        
        update_Q_value(SavedState(ImmutableMatrix(prev_s)), prev_a, reward, SavedState(ImmutableMatrix(curr_s)))
        
        steps = steps + 1
        epsilon = epsilon - 0.001
        #print("utility:")
        #print(utility)

        
    print("steps:")
    print(steps)
    print("Utility value: ")
    print(utility)
    print(list(Q_table.items()))


