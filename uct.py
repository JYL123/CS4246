import math
import cv2
import numpy as np
import gym
from copy import copy


# set up the env
env = gym.make('Centipede-ram-v0')
default_value = 0
# 4 actions from the action space
action_value = dict.fromkeys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], default_value)
action_times = dict.fromkeys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], default_value)
c = 0.6 # for UCT

print(env.action_space)

# run game for total steps
total = 200
for episode in range(total):
    # initialize the environment
    i_observation = env.reset()
    utility = 0

    steps = 20 # each game goes 200 steps
    for step in range(steps):
        # current state
        # no render() because visualization is not available on the cluster
        # env.render()
        # print(i_observation) # observation of the env, values being the pixel intensity of the env
        # Show the observation using OpenCV
        # cv2.imshow('obs', i_observation)
        # cv2.waitKey(1)

        # evaluate actions
        # sample k number times of actions, run trials to evaluate them
        k = 30
        # save state to be restored
        save_state = copy(env)

        for i in range(k):
            # env.render()
            # print("sample actions to run simulation: ")
            action = env.action_space.sample()
            # print(action)
            # increase the count of the action
            action_times[action] = action_times[action] + 1
            # look ahead 100 steps for each sampled action
            trial_steps = 20
            trial_reward = 0
            trial_utility = 0

            for trial_step in range(trial_steps):
                # print("running simulation: ")

                # env.env.restore_state(save_state)
                prev_observation = i_observation

                # choose actions with best values in trials
                # ran_action = env.action_space.sample()
                try_action = 0
                for key in action_value:
                    if action_value[key] > action_value[try_action]:
                        try_action = key

                # run trial with the chosen action
                # return these 4 variables after the action is taken
                curr_observation, reward, done, info = save_state.step(try_action)
                # reward received: number of blocks being hit, reward = number of 0 increased
                # m1 = np.count_nonzero(prev_observation)
                # m2 = np.count_nonzero(curr_observation)
                # trial_reward = m2 - m1
                trial_utility = trial_utility + reward

                prev_observation = curr_observation

            # calculate UCT
            # the value will be over-written several times, instead of being updated?
            # action_times[action]+1 to make sure it is not 0
            action_value[action] = (trial_utility + action_value[action]*(action_times[action]-1))/action_times[action] + c*math.sqrt(math.log(i+1)/(action_times[action]+1))
            print("updated value: ")
            print(action_value[action])

        # env.env.restore_state(save_state)
        # choose the best action
        max_action = 0
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
