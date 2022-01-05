"""
In this script, we will use native Top-down renderer to visualize and provide
observation. Panda3D rendering is avoided.
We will use the multichannel top-down images as observation.
"""
import argparse
import random

import numpy as np

from metadrive import MetaDriveEnv
from metadrive.constants import HELP_MESSAGE

if __name__ == "__main__":
    config = dict(
        # controller="joystick",
        use_render=False,
        manual_control=True,
        traffic_density=0.1,
        environment_num=100,
        random_agent_model=True,
        random_lane_width=True,
        random_lane_num=True,
        map=4,  # seven block
        start_seed=random.randint(0, 1000)
    )
    env = MetaDriveEnv(config)
    print(HELP_MESSAGE)

    o = env.reset()

    env.vehicle.expert_takeover = True
    print("The observation is an numpy array with shape: ", o.shape)

    for i in range(1, 1000000000):
        o, r, d, info = env.step([0, 0])
        env.render(mode="top_down")
        if d:
            env.reset()
            env.vehicle.expert_takeover = True
