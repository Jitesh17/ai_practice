import gym
from gym import wrappers
import numpy as np
env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, 'random_files', force=True)
for i_episode in range(1):
    observation = env.reset()
    for t in range(100000):
        env.render()
        print(observation)
        action = env.action_space.sample()
        action = np.random.randint(0,2)
        observation, reward, done, info = env.step(action)
        print(f'reward :{reward}')
        print(f'done :{done}')
        print(f'info :{info}')
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()