#!/usr/bin/env python
"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""
import argparse

from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.policy.replay_policy import ReplayEgoCarPolicy, WayPointPolicy
import numpy as np
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
    parser.add_argument("--add_sensor", action="store_true")
    args = parser.parse_args()
    extra_args = dict(film_size=(2000, 2000)) if args.top_down else {}
    asset_path = AssetLoader.asset_path
    use_waymo = args.waymo
    print(HELP_MESSAGE)

    cfg = {
        "agent_policy": WayPointPolicy,
        "map_region_size": 1024,  # use a large number if your map is toooooo big
        "sequential_seed": True,
        "reactive_traffic": True if args.reactive_traffic else False,
        "use_render": True if not args.top_down else False,
        "data_directory": AssetLoader.file_path(asset_path, "waymo" if use_waymo else "nuscenes", unix_style=False),
        "num_scenarios": 3 if use_waymo else 10,
    }
    if args.add_sensor:
        additional_cfg = {
            "interface_panel": ["rgb_camera", "depth_camera", "semantic"],
            "sensors": {
                "rgb_camera": (DepthCamera, 256, 256),
                "depth_camera": (RGBCamera, 256, 256),
                "semantic": (SemanticCamera, 256, 256)
            }
        }
        cfg.update(additional_cfg)

    try:
        env = ScenarioEnv(cfg)
        o, _ = env.reset()
        control_duration = 10 # 1 steps, equivalent to 2 seconds in wall time
        for i in range(1, 100000):
            # action = None will not modify the WaypointPolicy.online_traj_info
            # action in the following format will overwrite the trajectory.
            # Note that all these spatial information use ego coordinate, at 10HZ frequency.
            # velocity (10, 1) m/s, go front and go left
            # You can write as much waypoints as you wnat, as long as it's a np.array of shape (N,2)
            action = dict(
                position=np.array([
                    [i,0.01*i] for i in range(1, 11)
                ]),
            )
            o, r, tm, tc, info = env.step(actions=action)
            env.render(
                mode="top_down" if args.top_down else None,
                text=None if args.top_down else RENDER_MESSAGE,
                **extra_args
            )
            if tm or tc:
                env.reset()
    finally:
        env.close()
