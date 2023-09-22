"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""
import pygame
import os
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy


def test_top_down_semantics(render=False):
    dataset = "waymo"
    try:
        env = WaymoEnv(
            {
                "manual_control": False,
                "reactive_traffic": False,
                "no_light": True,
                "agent_policy": ReplayEgoCarPolicy,
                "use_render": False,
                "sequential_seed": True,
                "data_directory": "/home/shady/data/scenarionet/dataset/{}".format(dataset),
                "num_scenarios": 100
            }
        )
        o, _ = env.reset()
        for seed in range(100):
            env.reset(seed=seed)
            # dir_p = "{}_{}".format(dataset, seed)
            # os.makedirs(dir_p)
            for step in range(0, 1000):
                o, r, tm, tc, info = env.step([1.0, 0.])
                this_frame_fig = env.render(
                    mode="top_down",
                    semantic_map=True,
                    semantic_broken_line=False,
                    film_size=(8000, 8000),
                    num_stack=1,
                    draw_contour=False,
                    scaling=10,
                )
                # pygame.image.save(this_frame_fig, os.path.join(dir_p, "{}.png".format(step)))
                if tm:
                    break
    finally:
        env.close()


if __name__ == '__main__':
    test_top_down_semantics(True)
