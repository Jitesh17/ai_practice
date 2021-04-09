import os, sys
import time
import gym
from gym import wrappers
import numpy as np
import json
env = gym.make('SpaceInvaders-v0')
env_names = [
    'Acrobot-v1', 'CartPole-v1', 'MountainCar-v0', 
    'MountainCarContinuous-v0', 'Pendulum-v0'
             ]
env = gym.make('Pendulum-v0')
# env = wrappers.Monitor(env, 'random_files', force=True)

print(env.action_space)
# print(env.action_space.n)
print(env.action_space.high)
print(env.action_space.low)
print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)
a = gym.spaces.Box(0, 1, shape=(2,1), dtype=np.int32)
print(gym.spaces.Box(0, 1, shape=(2,1), dtype=np.int32))
print(a.high)
print(a.low)
# print(env.reset())
# print((env.observation_space.high-env.observation_space.low)/50)
observation = env.reset()
sys.exit()
for t in range(100000):
    env.render()
    # print(observation)
    # action = env.action_space.sample()
    action = np.random.randint(0,4)
    print(action)
    observation, reward, done, info = env.step(action)
    time.sleep(1)
    if done:
        break
sys.exit()
for i_episode in range(1):
    observation = env.reset()
    for t in range(100000):
        env.render()
        print(observation)
        # action = env.action_space.sample()
        action = np.random.randint(0,2)
        observation, reward, done, info = env.step(action)
        print(f'reward :{reward}')
        print(f'done :{done}')
        print(f'info :{info}')
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()