#!/usr/bin/env python
"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""
import numpy as np
from metadrive.component.sensors.point_cloud_lidar import PointCloudLidar
from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from panda3d.core import Point3
from metadrive.envs.scenario_env import ScenarioEnv

RENDER_MESSAGE = {
    "Quit": "ESC",
    "Switch perspective": "Q or B",
    "Reset Episode": "R",
    "Keyboard Control": "W,A,S,D",
}

if __name__ == "__main__":
    asset_path = AssetLoader.asset_path
    print(HELP_MESSAGE)
    try:
        env = ScenarioEnv(
            {
                "manual_control": True,
                "sequential_seed": True,
                "reactive_traffic": False,
                "use_render": True,
                "image_observation": True,
                "vehicle_config": dict(image_source="point_cloud"),
                "sensors": dict(point_cloud=(PointCloudLidar, 200, 64, True)),  # 64 channel lidar
                "data_directory": AssetLoader.file_path(asset_path, "nuscenes", unix_style=False),
                "num_scenarios": 10
            }
        )
        o, _ = env.reset()

        cam = env.engine.get_sensor("point_cloud").cam
        drawer = env.engine.make_line_drawer()
        for i in range(1, 100000):
            o, r, tm, tc, info = env.step([1.0, 0.])
            points = o["image"][..., :, -1] + np.asarray(env.engine.render.get_relative_point(cam, Point3(0, 0, 0)))

            drawer.reset()
            drawer.draw_lines(points)
            env.render(text=RENDER_MESSAGE, )
            if tm or tc:
                env.reset()
    finally:
        env.close()
