# THIS DOES NOT WORK THE TUTORIAL IS COMPLETELY OUT OF DATE FOR THIS


import gym
import numpy as np
import time 

env = gym.make('FrozenLake-v1',  render_mode='human') # we are going to use the FrozenLake environment

'''
print(env.observation_space.n) # get number of states
print(env.action_space.n) # get number of actions

env.reset() # reset environment to default size
action = env.action_space.sample() # get a random action
observation, reward, truncated, done, info = env.step(action) # take action, notice it returns information about the 
env.render() # render the GUI for the enviornment
'''

# Building the Q table
STATES = env.observation_space.n
ACTIONS = env.action_space.n

Q = np.zeros((STATES, ACTIONS)) # create a matrix with all 0 values (16x4 2d array)

EPISODES = 1500 # how many times to run the environment from the beginning
MAX_STEPS = 100 # max number of steps allowed for each run of environment
LEARNING_RATE = 0.81 # learning rate
GAMMA = 0.96

# Picking an Action
epsilon = 0.9 # start with a 90% chance of picking a random action

RENDER = False # change to true to see rendering

# code to pick action
rewards = []
for episode in range(EPISODES):
    state = env.reset()
    for _ in range(MAX_STEPS):
        if RENDER:
            env.render()
        
        if np.random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        next_state, reward, done, _, extra = env.step(action)
        Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state

        if done:
            rewards.append(reward)
            epsilon -= 0.001
            break # reached goal

print(Q)
print(f"Average reward: {sum(rewards)/len(rewards)}:") # and now we can see our Q values


# if np.random.uniform(0, 1) < epsilon: # we will check if a randomly selected value is less than epsilon
#     action = env.action_space.sample() # take random action
# else:
#     action = np.argmax(Q[state, :]) # use Q table to pick best action based on current values

# # Updating Q Values
# Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[new_state, :]) - Q[state, action])