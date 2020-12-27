"""
Please feel free to run this script to enjoy a journey carrying out by a professional driver!
Our expert can drive in 10000 maps with almost 90% likelihood to achieve the destination.

Note: This script require rendering, please following the installation instruction to setup a proper
environment that allows popping up an window.
"""
import random

from pgdrive import PGDriveEnv
from pgdrive.examples import expert, get_terminal_state

env = PGDriveEnv(dict(use_render=True, environment_num=100, start_seed=random.randint(0, 1000)))
obs = env.reset()
success_list, reward_list, ep_reward, ep_len, ep_count = [], [], 0, 0, 0
try:
    while True:
        action = expert(obs)
        obs, reward, done, info = env.step(action)
        ep_reward += reward
        ep_len += 1
        env.render()
        if done:
            ep_count += 1
            success_list.append(1 if get_terminal_state(info) == "Success" else 0)
            reward_list.append(ep_reward)
            print(
                "{} episodes terminated! Length: {}, Reward: {:.4f}, Terminal state: {}.".format(
                    ep_count, ep_len, ep_reward, get_terminal_state(info)
                )
            )
            ep_reward = 0
            ep_len = 0
            obs = env.reset()
finally:
    print("Closing the environment!")
    env.close()
    print(
        "Episode count {}, Success rate: {}, Average reward: {}".format(
            ep_count,
            sum(success_list) / len(success_list),
            sum(reward_list) / len(reward_list)
        )
    )
