#!/usr/bin/env python
import random

import matplotlib.pyplot as plt

from metadrive import MetaDriveEnv
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.utils.draw_top_down_map import draw_top_down_map

if __name__ == '__main__':
    env = MetaDriveEnv(config=dict(num_scenarios=100, map=7, start_seed=0))
    fig, axs = plt.subplots(2, 3, figsize=(10, 10), dpi=100)
    count = 0
    print("We are going to draw 6 maps! 3 for PG maps, 3 for real world ones!")
    for i in range(2):
        if i == 1:
            env.close()
            env = ScenarioEnv(dict(start_scenario_index=0, num_scenarios=3))
        for j in range(3):
            count += 1
            env.reset(seed=j)
            m = draw_top_down_map(env.current_map)
            # m = env.get_map()
            ax = axs[i][j]
            ax.imshow(m, cmap="bone")
            ax.set_xticks([])
            ax.set_yticks([])
            print("Drawing {}-th map!".format(count))
    fig.suptitle("Top-down view of generated maps")
    plt.show()
    env.close()
