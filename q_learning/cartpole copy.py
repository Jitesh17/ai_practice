import argparse
import json
import math
import os
import sys

import gym
import matplotlib.pyplot as plt
import numpy as np

from utils import Agent, Q, Trainer

RECORD_PATH = os.path.join(os.path.dirname(__file__), "./upload")


def main(episodes, render, monitor):
    episodes=2000
    env_type = 'MountainCar-v0'
    env = gym.make(env_type) 
    env = gym.wrappers.Monitor(env, "recording_"+env_type,force=True,
                               resume=True,
                            #    write_upon_reset=True, 
                               uid='j',mode='training')

    q = Q(
        env.action_space.n, 
        env.observation_space, 
        bin_size=[3, 3, 8, 5],
        low_bound=[None, -0.5, None, -math.radians(50)], 
        high_bound=[None, 0.5, None, math.radians(50)],
        initial_mean=10
        )
    agent = Agent(q, epsilon=0.05)
    s = 100*4
    learning_decay = lambda lr, t: max(0.1, min(1., lr - math.log10((t + 1) / s)))
    learning_decay = lambda lr, t: max(0.001, min(.1, lr*np.math.exp(-0.01)))
    epsilon_decay = lambda eps, t: max(0.01, min(1.0, eps*np.math.exp(-0.01)))
    learning_decay = lambda lr, t: max(0.01, min(lr, lr - math.log10((t + 1) / 300)/200))
    epsilon_decay = lambda eps, t: max(0.01, min(eps, eps - math.log10((t + 1) / 100)/200))
    # print(f'learning_decay(2): {learning_decay(2)}')
    x=[]
    y=[]
    z=[]
    yi = 1.
    zi = 1.
    for i in range(episodes):
        x.append(i)
        yi = learning_decay(yi,i)
        y.append(yi)
        zi = epsilon_decay(zi,i)
        z.append(zi)
    # plt.figure
    plt.plot(x, y, label = "learning_decay") 
    # plt.show() 
    plt.plot(x, z, label = "epsilon_decay") 
    plt.legend() 
    plt.show() 
    # return
    trainer = Trainer(
        agent, 
        gamma=0.99,
        learning_rate=0.5, learning_rate_decay=learning_decay, 
        epsilon=1.0, epsilon_decay=epsilon_decay,
        max_step=250)

    if monitor:
        env.monitor.start(RECORD_PATH)

    agent = trainer.train(env, episode_count=episodes, render=render, )
    
    env.close()
    env = gym.make(env_type) 
    env = gym.wrappers.Monitor(env, "recording_e_"+env_type,force=True,
                               resume=True,
                            #    write_upon_reset=True, 
                               uid='e',mode='evaluation')
    agent.epsilon = 0.01
    
    observation = env.reset()
    trainer.eval(env, render=render)
    # for t in range(100000):
    #     env.render()
    #     # print(observation)
    #     action = agent.act(observation)
    #     observation, reward, done, info = env.step(action)
    #     # print(f'reward :{reward}')
    #     # print(f'done :{done}')
    #     # print(f'info :{info}')
    #     if done:
    #         print("Episode finished after {} timesteps".format(t+1))
    #         break
    env.close()
    print(dict(agent.q.table))
    # with open('q_table.json', 'w') as outfile:
    #     json.dump(dict(agent.q.table), outfile, indent=4)
    if monitor:
        env.monitor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train & run cartpole ")
    parser.add_argument("--episode", type=int, default=2000//2, help="episode to train")
    parser.add_argument("--render", action="store_true", help="render the screen")
    parser.add_argument("--monitor", action="store_true", help="monitor")
    parser.add_argument("--upload", type=str, default="", help="upload key to openai gym (training is not executed)")

    args = parser.parse_args()

    if args.upload:
        if os.path.isdir(RECORD_PATH):
            gym.upload(RECORD_PATH, api_key=args.upload)
    else:
        main(args.episode, args.render, args.monitor)
