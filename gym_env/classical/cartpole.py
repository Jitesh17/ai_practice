import imageio
import gym
import numpy as np
from pyjeasy.image_utils import show_image
from stable_baselines3 import A2C
import os

current_dirname = os.path.dirname(__file__)
env_name = 'CartPole-v1'
env = gym.make(env_name)

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100_000)
model.save(f'{current_dirname}/{env_name}_a2c_01')
obs = model.env.reset()
images = []
img = model.env.render(mode='rgb_array')
for i in range(1000):
    images.append(img)
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = model.env.step(action)
    img = model.env.render(mode='rgb_array')
    show_image(img, f"step {i}", 800)
    # env.render()
    if done:
      obs = model.env.reset()
imageio.mimsave(f'{current_dirname}/{env_name}_a2c.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)