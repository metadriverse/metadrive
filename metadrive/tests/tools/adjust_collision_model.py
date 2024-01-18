import numpy as np
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.sensors.mini_map import MiniMap
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.dashboard import DashBoard

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
            "manual_control": True,
            "use_render": True,
            "decision_repeat": 5,
            "interface_panel": [MiniMap, DashBoard, RGBCamera],
            "need_inverse_traffic": False,
            "norm_pixel": True,
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
    o, _ = env.reset()

    def get_v_path():
        return BaseVehicle.model_collection[env.agent.path[0]]

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
        o, r, tm, tc, i = env.step([0, 0])
        env.render(
            text={
                "heading_diff": env.agent.heading_diff(env.agent.lane),
                "lane_width": env.agent.lane.width,
                "lateral": env.agent.lane.local_coordinates(env.agent.position),
                "current_seed": env.current_seed
            }
        )
