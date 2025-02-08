#!/usr/bin/env python
"""
This file illustrate how to use top-down renderer to provide observation in form of multiple channels of semantic maps.

We let the target vehicle moving forward directly. You can also try to control the vehicle by yourself. See the config
below for more information.

This script will popup a Pygame window, but that is not the form of the observation. We will also popup a matplotlib
window, which shows the details observation of the top-down pygame renderer.

The detailed implementation of the Pygame renderer is in TopDownMultiChannel Class (a subclass of Observation Class)
at: metadrive/obs/top_down_obs_multi_channel.py

We welcome contributions to propose a better representation of the top-down semantic observation!
"""

import random

import matplotlib.pyplot as plt

from metadrive.envs.top_down_env import TopDownMetaDrive
from metadrive.constants import HELP_MESSAGE
from metadrive.examples.ppo_expert.numpy_expert import expert


def draw_multi_channels_top_down_observation(obs, show_time=4):
    num_channels = obs.shape[-1]
    assert num_channels == 5
    channel_names = [
        "Road and navigation", "Ego now and previous pos", "Neighbor at step t", "Neighbor at step t-1",
        "Neighbor at step t-2"
    ]
    fig, axs = plt.subplots(1, num_channels, figsize=(15, 4), dpi=80)
    count = 0

    def close_event():
        plt.close()  # timer calls this function after 3 seconds and closes the window

    timer = fig.canvas.new_timer(
        interval=show_time * 1000
    )  # creating a timer object and setting an interval of 3000 milliseconds
    timer.add_callback(close_event)

    for i, name in enumerate(channel_names):
        count += 1
        ax = axs[i]
        ax.imshow(obs[..., i], cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)
        # print("Drawing {}-th semantic map!".format(count))
    fig.suptitle("Multi-channels Top-down Observation")
    timer.start()
    plt.show()


if __name__ == "__main__":
    print(HELP_MESSAGE)
    env = TopDownMetaDrive(
        dict(
            # We also support using two renderer (Panda3D renderer and Pygame renderer) simultaneously. You can
            # try this by uncommenting next line.
            # use_render=True,

            # You can also try to uncomment next line with "use_render=True", so that you can control the ego vehicle
            # with keyboard in the main window.
            # manual_control=True,
            map="SSSS",
            traffic_density=0.1,
            num_scenarios=100,
            start_seed=random.randint(0, 1000),
        )
    )
    try:
        o, _ = env.reset()
        for i in range(1, 100000):
            o, r, tm, tc, info = env.step(expert(env.agent))
            env.render(mode="top_down", text={"Quit": "ESC"}, film_size=(2000, 2000))
            if tm or tc:
                env.reset()
            if i % 50 == 0:
                draw_multi_channels_top_down_observation(o, show_time=5)  # show time 4s
                # ret = input("Do you wish to quit? Type any ESC to quite, or press enter to continue")
                # if len(ret) == 0:
                #     continue
                # else:
                #     break
    finally:
        env.close()
