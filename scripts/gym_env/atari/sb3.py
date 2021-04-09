import os

import imageio
import numpy as np
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.ve
German Aerospace Center (DLR) - Instc_env import VecFrameStack
from utils import image_with_episode_number

current_dirname = os.path.dirname(__file__)
# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
algo_name = 'ppo'
env_name = 'SpaceInvaders-v0'
n = 1
env = make_atari_env(env_name, n_envs=n, seed=0)
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=n)

model = PPO(
    'CnnPolicy', env, verbose=1,
    tensorboard_log=f'{current_dirname}/tb/{env_name}',
    
                )

# model = PPO.load(f'{current_dirname}/models/{env_name}_{algo_name}_01', env)
model.learn(total_timesteps=100_000)
model.save(f'{current_dirname}/models/{env_name}_{algo_name}_02')
    
obs = model.env.reset()
images = []
img = model.env.render(mode='rgb_array')
# while True:
for i in range(1000):
    # images.append(image_with_episode_number(img, step_num=i))
    images.append(img)
        
    action, _states = model.predict(obs)
    try:
        obs, rewards, done, info = model.env.step(action)
    except:
        obs = model.env.reset()
        continue
    try:
        print(i, action, obs.shape, rewards, done)
    except:
        pass
    img = model.env.render(mode='rgb_array')
        
    if any(done):
        break
        obs = model.env.reset()
    # model.env.render()
imageio.mimsave(f'{current_dirname}/gifs/{env_name}_{algo_name}_3.gif',
                    [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=30)

"""
def test(render: bool=False, samples: int=10_000):
    observation = env.reset()
    total_reward = 0
    for t in range(samples):
        if render:
            env.render()
        # action = env.action_space.sample()
        # action = np.random.randint(0,2
        # )
        action, _ = model.predict(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        # print(f'{t} action: {action} reward: {total_reward} done: {done}')
        # time.sleep(1)
        if done:
            break
    return total_reward

scores=[]
for i in range(100):
    scores.append(test())
avg_score = sum(scores)/len(scores)
print(avg_score)

"""