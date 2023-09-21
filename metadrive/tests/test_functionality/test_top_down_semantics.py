"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""
import argparse
import random

import pygame

from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv


class DemoWaymoEnv(WaymoEnv):
    def reset(self, seed=None):
        if self.engine is not None:
            seeds = [i for i in range(self.config["num_scenarios"])]
            seeds.remove(self.current_seed)
            seed = random.choice(seeds)
        return super(DemoWaymoEnv, self).reset(seed=seed)


def test_top_down_semantics(render=False):
    asset_path = AssetLoader.asset_path
    try:
        env = DemoWaymoEnv(
            {
                "manual_control": True,
                "reactive_traffic": False,
                "use_render": False,
                "data_directory": AssetLoader.file_path(asset_path, "nuscenes", return_raw_style=False),
                "num_scenarios": 3
            }
        )
        o, _ = env.reset()

        for i in range(1, 1000):
            o, r, tm, tc, info = env.step([1.0, 0.])
            if render:
                this_frame_fig = env.render(
                    mode="top_down",
                    semantic_map=True,
                    film_size=(5000, 5000),
                    num_stack=1,
                    # scaling=10,
                )
            # save
            # pygame.image.save(this_frame_fig, "{}.png".format(i))
            if tm or tc:
                env.reset()
    finally:
        env.close()


if __name__ == '__main__':
    test_top_down_semantics(True)
