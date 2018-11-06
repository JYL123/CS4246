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
N_table = defaultdict(float)
# others 
utility = 0 
steps  = 0
gamma= 0.99
epsilon = 0.1
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_next', 'done'])


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

def update_Q_value(Transit):
    # previous_state, previous_action, previous_reward, current_state
    max_q_next = max([Q_table[Transit.s_next, a] for a in range(env.action_space.n)])
    # We do not include the value of the next state if terminated.
    Q_table[Transit.s, Transit.a] += (1/N_table[Transit.s, Transit.a]) * (
        Transit.r + gamma * max_q_next  - Q_table[Transit.s, Transit.a])

print(Q_table)

# run game for a number of runs
samples = 23
for sameple in range(samples):
    # initialize the environment
    curr_s = env.reset()
    curr_r = 0
    done = False

    while not done:
        prev_a = epsilon_greedy(curr_s)
        print(prev_a)
        next_state, reward, done, info = env.step(prev_a)
        if reward is not None:
            prev_r = reward
            utility = utility + reward
        
        prev_s = curr_s
        curr_s = next_state
        print(curr_s)
        
        update_Q_value(Transition(prev_s, prev_a, prev_r, curr_s, done))
        
        steps = steps + 1
        epsilon = epsilon - 0.001

print("Numbere of steps for the game:")
print(steps)
print("Utility value: ")
print(utility)

