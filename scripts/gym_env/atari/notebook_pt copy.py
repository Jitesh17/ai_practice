# In[]:
import gym
from kaggle_environments import make, evaluate

import os
import numpy as np
import torch as th
from torch import nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from submission_random2 import agent as agent_r

# # %%
# # ConnectX wrapper from Alexis' notebook.
# # Changed shape, channel first.
# # Changed obs/2.0
# class ConnectFourGym(gym.Env):
#     def __init__(self, agent2=agent_r):
#         ks_env = make("connectx", debug=True)
#         self.env = ks_env.train([None, agent2])
#         self.rows = ks_env.configuration.rows
#         self.columns = ks_env.configuration.columns
#         # Learn about spaces here: http://gym.openai.com/docs/#spaces
#         self.action_space = gym.spaces.Discrete(self.columns)
#         self.observation_space = gym.spaces.Box(low=0, high=1, 
#                                             shape=(1,self.rows,self.columns), dtype=np.float)
#         # Tuple corresponding to the min and max possible rewards
#         self.reward_range = (-10, 1)
#         # StableBaselines throws error if these are not defined
#         self.spec = None
#         self.metadata = None
#     def reset(self):
#         self.obs = self.env.reset()
#         return np.array(self.obs['board']).reshape(1,self.rows,self.columns)/2
#     def change_reward(self, old_reward, done):
#         if old_reward == 1: # The agent won the game
#             return 1
#         elif done: # The opponent won the game
#             return -1
#         else: # Reward 1/42
#             return 1/(self.rows*self.columns)
#     def step(self, action):
#         # Check if agent's move is valid
#         is_valid = (self.obs['board'][int(action)] == 0)
#         if is_valid: # Play the move
#             self.obs, old_reward, done, _ = self.env.step(int(action))
#             reward = self.change_reward(old_reward, done)
#         else: # End the game and penalize agent
#             reward, done, _ = -10, True, {}
#         return np.array(self.obs['board']).reshape(1,self.rows,self.columns)/2, reward, done, _
# # %%
# env = ConnectFourGym()
# env
# %%
env = gym.make('SpaceInvaders-v0')
# %%
log_dir = "log/"
os.makedirs(log_dir, exist_ok=True)

# Logging progress
env = Monitor(env, log_dir, allow_early_resets=True)
env
# %%
env = DummyVecEnv([lambda: env])
env
# %%
env.observation_space.sample()
# %%
class Net(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512//2):
        super(Net, self).__init__(observation_space, features_dim)
        inp_size = 256
        self.transform = transforms.Compose([transforms.Resize((inp_size, inp_size))])
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        f1 = np.prod([32, inp_size//4, inp_size//4])
        self.fc3 = nn.Linear(f1, f1//4)
        self.fc4 = nn.Linear(f1//4, features_dim)

    def forward(self, x):
        x = self.transform(x)  # 3, 256, 256
        x = F.relu(self.conv1(x))  # 16, 256, 256
        x = self.pool(x)  # 16, 128, 128
        x = F.relu(self.conv2(x))  # 32, 128, 128
        x = self.pool(x)  # 32, 64, 64
        x = nn.Flatten()(x)  # 
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x
# %%
# net = Net(env.observation_space)
# transform = transforms.Compose([transforms.Resize((256, 256))])
# x = torch.Tensor(1, 3, 210, 160)
# x = net.transform(x)
# x = F.relu(net.conv1(x))
# x = net.pool(x)
# x = F.relu(net.conv2(x))
# x = net.pool(x)
# x.shape
# %%
policy_kwargs = {
    'activation_fn':th.nn.ReLU, 
    'net_arch':[64, dict(pi=[32, 16], vf=[32, 16])],
    'features_extractor_class':Net,
}
# learner = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs)
learner = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs)

learner.policy
# %%
# %%time
# learner.load("SpaceInvaders_ppo_01")
learner.learn(total_timesteps=1_00, )
learner.save("SpaceInvaders_ppo_02")
# %%
df = load_results(log_dir)['r']
df.rolling(window=10).mean().plot()
# %%
action, _ = learner.predict(env.reset())
# action[0]
type(action[0])
# type(int(action[0][0]))
# type(env.action_space.sample())
# %%
# env = gym.make('SpaceInvaders-v0')
def test(render: bool=False, samples: int=10_000):
    observation = env.reset()
    total_reward = 0
    for t in range(samples):
        if render:
            env.render()
        # action = env.action_space.sample()
        # action = np.random.randint(0,2
        # )
        action, _ = learner.predict(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        # print(f'{t} action: {action} reward: {total_reward} done: {done}')
        # time.sleep(1)
        if done:
            break
    return total_reward
# %%
scores=[]
for i in range(10):
    scores.append(test())
avg_score = sum(scores)/len(scores)
print(avg_score)

# %%

# %%
