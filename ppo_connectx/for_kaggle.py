# In[]:
!apt-get update -qq
!apt-get install -qq -y cmake libopenmpi-dev python3-dev zlib1g-dev
!python -m pip install -q --upgrade pip
!pip install -q --upgrade kaggle-environments
!pip install -q 'tensorflow==1.15.0'
!pip install -q 'stable-baselines[mpi]==2.10.0'

# In[]:
# import os
import random

# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# from gym import spaces
# from kaggle_environments import evaluate, make
from stable_baselines import PPO1

# from stable_baselines.a2c.utils import conv, conv_to_fc, linear
# from stable_baselines.bench import Monitor
# from stable_baselines.common.policies import CnnPolicy
# from stable_baselines.common.vec_env import DummyVecEnv

model = PPO1.load("ppo1_connectx_1")

# In[]:
def agent1(obs, config):
    # Use the best model to select a column
    col, _ = model.predict(np.array(obs['board']).reshape(6, 7, 1))
    # Check if selected column is valid
    is_valid = (obs['board'][int(col)] == 0)
    # If not valid, select random move.
    if is_valid:
        return int(col)
    else:
        return random.choice([col for col in range(config.columns) if obs.board[int(col)] == 0])
