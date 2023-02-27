import gym

env = gym.make('FrozenLake-v1') # we are going to use the FrozenLake environment

print(env.observation_space.n) # get number of states
print(env.action_space.n) # get number of actions

env.reset() # reset environment to default size
action = env.action_space.sample() # get a random action
observation, reward, done, info = env.step(action) # take action, notice it returns