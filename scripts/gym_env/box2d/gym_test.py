import json
import os
import sys
import time

import gym
import numpy as np
from gym import wrappers
from tqdm import tqdm

env_names = [
    'Copy-v0'
             ]
# env = wrappers.Monitor(env, 'random_files', force=True)

for env_name in tqdm(env_names):
    print("########################################")
    print(env_name)
    env = gym.make(env_name)
    print("action_space:", env.action_space)
    # try:
    #     print(env.action_space.n)
    # except:
    #     print(env.action_space.high)
    #     print(env.action_space.low)
    # else:
    #     pass
    print("observation_space:", env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)
# a = gym.spaces.Box(0, 1, shape=(2,1), dtype=np.int32)
# print(gym.spaces.Box(0, 1, shape=(2,1), dtype=np.int32))
# print(a.high)
# print(a.low)
# # print(env.reset())
# # print((env.observation_space.high-env.observation_space.low)/50)
# observation = env.reset()
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
