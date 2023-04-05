"""
Compared to top_down_metadrive.py, this scripts demo the BEV renderer that resembles NuScenes dataset.
"""
import random

import matplotlib.pyplot as plt

from metadrive import TopDownMetaDriveEnvV3
from metadrive.constants import HELP_MESSAGE
from metadrive.examples.ppo_expert.numpy_expert import expert


def draw_multi_channels_top_down_observation(obs, show_time=4):
    num_channels = obs.shape[-1]
    assert num_channels == 5
    channel_names = [
        "driveable_area", "lane_lines", "actors"
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
    fig.suptitle("NuScenes BEV Observation")
    timer.start()
    plt.show()


if __name__ == "__main__":
    print(HELP_MESSAGE)
    env = TopDownMetaDriveEnvV3(
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
        o = env.reset()
        for i in range(1, 100000):
            o, r, d, info = env.step(expert(env.vehicle))
            env.render(mode="top_down", film_size=(800, 800))
            if d:
                env.reset()
            if i % 50 == 0:
                draw_multi_channels_top_down_observation(o, show_time=4)  # show time 4s
                # ret = input("Do you wish to quit? Type any ESC to quite, or press enter to continue")
                # if len(ret) == 0:
                #     continue
                # else:
                #     break
    except Exception as e:
        raise e
    finally:
        env.close()
