import math
import cv2
import numpy as np
import gym


# set up the env
env = gym.make('Breakout-ram-v0')
default_value = 0
# 4 actions from the action space
action_value = dict.fromkeys([0, 1, 2, 3], default_value)
action_times = dict.fromkeys([0, 1, 2, 3], default_value)
c = 0.2 # for UCT

# run game for total steps
total = 20
for episode in range(total):
    # initialize the environment
    i_observation = env.reset()
    # calculate rewards being the number of pixel changed 0
    cal_reward = 0

    steps = 20 # each game go 20 steps
    for step in range(steps):
        # current state
        env.render()
        print(i_observation) # observation of the env, values being the pixel intensity of the env
        # Show the observation using OpenCV
        # cv2.imshow('obs', i_observation)
        # cv2.waitKey(1)

        # choose action
        # sample k number times of actions, run trials to evaluate acitons
        k = 30
        for i in range(k):
            env.render()
            print("sample actions to run simulation: ")
            print(i)
            # sample an action to run trials
            action = env.action_space.sample()
            action_times[action] = action_times[action] + 1
            # print("action: ")
            # print(action)
            # look ahead 20 steps for each sampled aciton
            trial_steps = 50
            trial_reward = 0 # initial reward is 0
            trial_utility = 0
            for trial_step in range(trial_steps):
                print("running simulation: ")
                env.render()
                prev_observation = i_observation

                # choose actions with best values in trials
                # ran_action = env.action_space.sample()
                try_action = 0
                for key in action_value:
                    if action_value[key] > action_value[try_action]:
                        try_action = key

                # run trial with the chosen action
                curr_observation, reward, done, info = env.step(try_action)
                # reward received: number of blocks being hit, reward = number of 0 increased
                m1 = np.count_nonzero(prev_observation)
                m2 = np.count_nonzero(curr_observation)
                trial_reward = m2 - m1
                trial_utility = trial_utility + trial_reward

                prev_observation = curr_observation

            # calculate UCT
            # the value will be over-written several times, instead of being updated?
            # action_times[action]+1 to make sure it is not 0
            action_value[action] = trial_utility + c*math.sqrt(math.log(i+1)/(action_times[action]+1))

        # choose the best action
        max_action = 0
        for key in action_value:
            if action_value[key] > action_value[max_action]:
                max_action = key
        for key in action_value:
            print("action: ")
            print(key)
            print("value: ")
            print(action_value[key])
        print("best action: ")
        print(max_action)

        observation, reward, done, info = env.step(max_action)
        i_observation = observation # update observation
        print("reward: ")
        print(reward)
        print("info: ")
        print(info)

        if done:
            print("Episode finished after {} timesteps".format(episode+1))
            break
env.close()