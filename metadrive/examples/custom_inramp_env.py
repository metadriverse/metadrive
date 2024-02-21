#!/usr/bin/env python
"""
This script defines a custom environment with single block: inramp.
"""
import random

import numpy as np

from metadrive import MetaDriveEnv

if __name__ == "__main__":
    config = dict(
        use_render=True,
        manual_control=True,
        start_seed=random.randint(0, 1000),

        # Solution 1: use easy config to customize the map
        # map="r",  # seven block

        # Solution 2: you can define more complex map config
        map_config=dict(lane_num=1, type="block_sequence", config="r")
    )

    env = MetaDriveEnv(config)
    try:
        o, _ = env.reset()
        env.agent.expert_takeover = True
        assert isinstance(o, np.ndarray)
        print("The observation is an numpy array with shape: ", o.shape)
        for i in range(1, 1000000000):
            o, r, tm, tc, info = env.step([0, 0])
            if (tm or tc) and info["arrive_dest"]:
                env.reset()
                env.current_track_agent.expert_takeover = True
    finally:
        env.close()
