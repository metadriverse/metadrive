#!/usr/bin/env python
"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""
import argparse
import random

from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv

RENDER_MESSAGE = {
    "Quit": "ESC",
    "Switch perspective": "Q or B",
    "Reset Episode": "R",
    "Keyboard Control": "W,A,S,D",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reactive_traffic", action="store_true")
    parser.add_argument("--top_down", "--topdown", action="store_true")
    parser.add_argument("--waymo", action="store_true")
    args = parser.parse_args()
    extra_args = dict(film_size=(2000, 2000)) if args.top_down else {}
    asset_path = AssetLoader.asset_path
    use_waymo = args.waymo
    print(HELP_MESSAGE)
    try:
        env = ScenarioEnv(
            {
                "manual_control": True,
                "sequential_seed": True,
                "reactive_traffic": True if args.reactive_traffic else False,
                "use_render": True if not args.top_down else False,
                "data_directory": AssetLoader.file_path(
                    asset_path, "waymo" if use_waymo else "nuscenes", unix_style=False
                ),
                "num_scenarios": 3 if use_waymo else 10,
                "debug": True,
                "use_bounding_box": True,
                "vehicle_config": {
                    "show_line_to_dest": True,
                    "show_line_to_navi_mark": True,
                },
                "disable_collision": True,
            }
        )
        o, _ = env.reset()

        for i in range(1, 100000):
            o, r, tm, tc, info = env.step([1.0, 0.])
            env.render(
                mode="top_down" if args.top_down else None,
                text=None if args.top_down else RENDER_MESSAGE,
                **extra_args
            )
            if tm or tc:
                env.reset()
    finally:
        env.close()
