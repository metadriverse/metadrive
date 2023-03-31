import numpy as np
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.vehicle_module.mini_map import MiniMap
from metadrive.component.vehicle_module.rgb_camera import RGBCamera
from metadrive.component.vehicle_module.vehicle_panel import VehiclePanel

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils import setup_logger

if __name__ == "__main__":
    setup_logger(True)
    env = MetaDriveEnv(
        {
            "num_scenarios": 10,
            "traffic_density": 0.2,
            "traffic_mode": "hybrid",
            "start_seed": 22,
            "debug": True,
            "cull_scene": False,
            "manual_control": True,
            "use_render": True,
            "decision_repeat": 5,
            "interface_panel": [MiniMap, VehiclePanel, RGBCamera],
            "need_inverse_traffic": False,
            "rgb_clip": True,
            "map": "SSS",
            # "agent_policy": IDMPolicy,
            "random_traffic": False,
            "random_lane_width": True,
            "random_agent_model": True,
            "driving_reward": 1.0,
            "force_destroy": False,
            "show_interface": False,
            "vehicle_config": {
                "enable_reverse": False,
            },
        }
    )
    import time

    start = time.time()
    o = env.reset()

    def get_v_path():
        return BaseVehicle.model_collection[env.vehicle.path[0]]

    def add_x():
        model = get_v_path()
        model.setX(model.getX() + 0.1)
        # print(model.getPos())

    def decrease_x():
        model = get_v_path()
        model.setX(model.getX() - 0.1)
        # print(model.getPos())

    def add_y():
        model = get_v_path()
        model.setY(model.getY() + 0.1)
        # print(model.getPos())

    def decrease_y():
        model = get_v_path()
        model.setY(model.getY() - 0.1)
        # print(model.getPos())

    env.engine.accept("i", add_x)
    env.engine.accept("k", decrease_x)
    env.engine.accept("j", add_y)
    env.engine.accept("l", decrease_y)

    for s in range(1, 10000):
        o, r, d, info = env.step([0, 0])
        env.render(
            text={
                "heading_diff": env.vehicle.heading_diff(env.vehicle.lane),
                "lane_width": env.vehicle.lane.width,
                "lateral": env.vehicle.lane.local_coordinates(env.vehicle.position),
                "current_seed": env.current_seed
            }
        )
