from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils import setup_logger

if __name__ == "__main__":
    setup_logger(True)
    env = MetaDriveEnv(
        {
            "environment_num": 1,
            "traffic_density": 0.1,
            "traffic_mode": "hybrid",
            "start_seed": 5,
            # "debug_physics_world": True,
            "pstats": True,
            # "controller":"joystick",
            "manual_control": True,
            "use_render": True,
            "decision_repeat": 5,
            "rgb_clip": True,
            "debug": True,
            "map_config": {
                BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                BaseMap.GENERATE_CONFIG: "yY",
                BaseMap.LANE_WIDTH: 3.5,
                BaseMap.LANE_NUM: 3,
            },
            "driving_reward": 1.0,
            "vehicle_config": {
                "show_lidar": False,
                "show_side_detector": True,
                "show_lane_line_detector": True,
                "lane_line_detector": {
                    "num_lasers": 100
                }
            }
        }
    )

    o = env.reset()
    print("vehicle num", len(env.engine.traffic_manager.vehicles))
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        info["fuel"] = env.vehicle.energy_consumption
        env.render(
            text={
                "left": env.vehicle.dist_to_left_side,
                "right": env.vehicle.dist_to_right_side,
                "white_lane_line": env.vehicle.on_white_continuous_line
            }
        )
        if d:
            print("Reset")
            env.reset()
    env.close()
