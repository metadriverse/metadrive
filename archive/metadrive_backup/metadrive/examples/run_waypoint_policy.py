"""
This script demonstrates how to use the Waypoint Policy, which feeds (5, 2), that is 5 waypoints, to the ego agent.
The waypoint is in the local frame of the vehicle, where the x-axis is the forward direction of the vehicle and
the y-axis is the left direction of the vehicle.
"""
import argparse

import numpy as np

from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioWaypointEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_down", "--topdown", action="store_true")
    parser.add_argument("--waymo", action="store_true")
    args = parser.parse_args()
    extra_args = dict(film_size=(2000, 2000)) if args.top_down else {}
    asset_path = AssetLoader.asset_path
    use_waymo = args.waymo

    waypoint_horizon = 5

    cfg = {
        "map_region_size": 1024,  # use a large number if your map is toooooo big
        "sequential_seed": True,
        "use_render": False,
        "data_directory": AssetLoader.file_path(asset_path, "waymo" if use_waymo else "nuscenes", unix_style=False),
        "num_scenarios": 3 if use_waymo else 10,
        "waypoint_horizon": waypoint_horizon,
    }

    try:
        env = ScenarioWaypointEnv(cfg)
        o, _ = env.reset()
        i = 0
        for _ in range(0, 100000):
            if i % waypoint_horizon == 0:
                # X-coordinate is the forward direction of the vehicle, Y-coordinate is the left of the vehicle
                x_displacement = np.linspace(1, 6, waypoint_horizon)
                y_displacement = np.linspace(0, 0.05, waypoint_horizon)  # pos y is left.
                action = dict(position=np.stack([x_displacement, y_displacement], axis=1))

            else:
                action = None

            o, r, tm, tc, info = env.step(actions=action)
            env.render(mode="top_down")
            i += 1
            if tm or tc:
                env.reset()
                i = 0
    finally:
        env.close()
