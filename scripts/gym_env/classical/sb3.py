import os

import gym
import imageio
import numpy as np
from pyjeasy.image_utils import show_image
from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm

from utils import image_with_episode_number

env_names = [
    'Acrobot-v1', 'CartPole-v1', 'MountainCar-v0', 
    'MountainCarContinuous-v0', 'Pendulum-v0'
             ]

current_dirname = os.path.dirname(__file__)
algo_name = 'ppo'  # Just for naming the files
env_name = 'Copy-v0'

for i in [1]:
# for env_name in tqdm(env_names):
    env = gym.make(env_name)
    model = PPO(
        'MlpPolicy', env, verbose=1,
        # learning_rate = 0.0003, n_steps = 20480, 
        # batch_size = 64, n_epochs = 10, 
        # gamma = 0.99, gae_lambda = 0.95, 
        # clip_range = 0.2, clip_range_vf = None, 
        # ent_coef = 0.001, vf_coef = 0.5, max_grad_norm = 0.5, 
        # use_sde = False, sde_sample_freq = - 1, 
        # target_kl = 0.01, 
        tensorboard_log = f'{current_dirname}/tb/{env_name}', 
        # create_eval_env = False, policy_kwargs = None, seed = 1
        )
    
    """
    model_class = DQN
    goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

    # If True the HER transitions will get sampled online
    online_sampling = True
    
    model = HER('MlpPolicy', env, model_class, verbose=1,
                n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy, online_sampling=online_sampling,
                tensorboard_log=f'{current_dirname}/tb/{env_name}_01',
                max_episode_length=15)
    """
    # model = DDPG.load(f'{current_dirname}/models/{env_name}_{algo_name}_02')
    # model.set_env(env)
    model.learn(total_timesteps=200_000, reset_num_timesteps=True)
    model.save(f'{current_dirname}/models/{env_name}_{algo_name}_02')
    # model = HER.load(f'{current_dirname}/models/Acrobot-v1_a2c_01.zip', env)
    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=10)

    obs = model.env.reset()
    images = []
    img = model.env.render(mode='rgb_array')
    for i in range(300):
        images.append(image_with_episode_number(img, step_num=i))
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = model.env.step(action)
        img = model.env.render(mode='rgb_array')
        # show_image(img, f"step {i}", 800)
        # env.render()
        if done:
            obs = model.env.reset()
    imageio.mimsave(f'{current_dirname}/gifs/{env_name}_{algo_name}.gif',
                    [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=30)
