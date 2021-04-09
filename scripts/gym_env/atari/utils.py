import os
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
# import matplotlib.pyplot as plt    


def image_with_episode_number(frame, episode_num=None, step_num=None):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    text = ''
    if episode_num is not None:
        text += f'Episode: {episode_num+1} '
    if step_num is not None:
        text += f'Step: {step_num+1} '
    drawer.text((im.size[0]/20,im.size[1]/18), text, fill=text_color)

    return im


def save_random_agent_gif(env):
    frames = []
    for i in range(5):
        state = env.reset()        
        for t in range(500):
            action = env.action_space.sample()

            frame = env.render(mode='rgb_array')
            frames.append(image_with_episode_number(frame, episode_num=i))

            state, _, done, _ = env.step(action)
            if done:
                break

    env.close()

    imageio.mimwrite(os.path.join('./videos/', 'random_agent.gif'), frames, fps=60)
