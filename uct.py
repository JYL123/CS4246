import math
import cv2
import numpy as np
import gym
import copy
import multiprocessing as mp

# set up the env
env = gym.make('Centipede-ram-v0')
default_value = math.inf
default_times = 0
# 4 actions from the action space
action_value = dict.fromkeys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], default_value)
action_times = dict.fromkeys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], default_times)
c = 0.6 # for UCT

# check the number of actions available
# print(env.action_space)

def run_trials(saved, i_observation, action_Q, action_times, sameple_times):
    env.reset()
    env.env.restore_full_state(saved)

    # choose an with highest value: in the begining it is the action hasn't tried before to make sure exploration
    action = env.action_space.sample()
    for key in action_value:
        if action_value[key] > action_value[action]:
            action = key

    # increase the count of the action
    action_times[action] = action_times[action] + 1

    # look ahead 30 steps for each sampled action
    trial_steps = 30
    trial_reward = 0
    trial_utility = reward
    gamma = 0.5
    total_gamma = 1
    # look ahead a number of steps 
    for trial_step in range(trial_steps):
        # dicount
        total_gamma = total_gamma * gamma
        # env.env.restore_state(save_state)
        prev_observation = i_observation

        # random sample actions 
        try_action = env.action_space.sample()
        
        # run trials
        curr_observation, trial_reward, done, info = env.step(try_action)

        # update   
        trial_utility = trial_utility + total_gamma * trial_reward
        prev_observation = curr_observation
    
    # update action value immediately
    action_Q[action] = action_Q[action] + trial_utility
    # calculate UCT
    action_value[action] = (action_Q[action]/action_times[action] + c*math.sqrt(math.log(sameple_times+1)/(action_times[action]+0.0000000001)))

def parallel_run_trials(action_samples, saved, i_observation, action_Q, action_times):
    # max number of processes run at the same time
    pool = mp.Pool(30)
    [pool.apply_async(run_trials(saved, i_observation, action_Q, action_times, sameple_times)) for sameple_times in range(action_samples)]  
    # no more new processes
    pool.close() 
    # wait till all trails return as we are using async 
    pool.join()

# run game for total steps
total = 23
print("Number of CPU on this machine:")
print(mp.cpu_count())
for episode in range(total):
    # initialize the environment
    i_observation = env.reset()
    utility = 0
    reward = 0
    steps = 200 # each game goes 200 steps
    for step in range(steps):
        action_value = dict.fromkeys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], default_value)
        action_times = dict.fromkeys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], default_times)

        # save current state
        saved = env.env.clone_full_state()
        
        # evaluate actions
        # sample k number times of actions, run trials to evaluate them
        k = 100

        # record down the avergae value for each action simulated
        action_Q = dict.fromkeys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], 0)
        parallel_run_trials(k, saved, i_observation, action_Q, action_times)

        # exit from simulation of the node
        env.reset()
        env.env.restore_full_state(saved)
        # choose the best action
        max_action = 14
        for key in action_value:
           if action_value[key] > action_value[max_action]:
               max_action = key

        print("best action: ")
        print(max_action)
        print(action_value[max_action])

        observation, reward, done, info = env.step(max_action)
        i_observation = observation # update observation
        utility = utility + reward
        print("utility: ")
        print(utility)

        if done:
            print("Episode finished after {} time steps".format(episode+1))
            break
env.close()
