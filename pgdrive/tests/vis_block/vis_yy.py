from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.scene_creator.map import Map, MapGenerateMethod
from pgdrive.utils import setup_logger


class TestEnv(PGDriveEnv):
    def __init__(self):
        """
        TODO a small bug exists in scene 9 (30 blocks), traffic density > 0, reborn mode
        """
        super(TestEnv, self).__init__(
            {
                "environment_num": 1,
                "traffic_density": 0.1,
                "traffic_mode": "hybrid",
                "start_seed": 5,
                "pg_world_config": {
                    "onscreen_message": True,
                    # "debug_physics_world": True,
                    "pstats": True
                },
                # "controller":"joystick",
                "manual_control": True,
                "use_render": True,
                "decision_repeat": 5,
                "rgb_clip": True,
                "debug": True,
                "map_config": {
                    Map.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                    Map.GENERATE_CONFIG: "yY",
                    Map.LANE_WIDTH: 3.5,
                    Map.LANE_NUM: 3,
                },
                "driving_reward": 1.0,
                "vehicle_config": {
                    "show_lidar": False,
                    "show_side_detector": True,
                    "show_lane_line_detector": True,
                    "use_lane_line_detector": True,
                }
            }
        )


if __name__ == "__main__":
    setup_logger(True)
    env = TestEnv()

    o = env.reset()
    print("vehicle num", len(env.scene_manager.traffic_mgr.vehicles))
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        info["fuel"] = env.vehicle.energy_consumption
        env.render(
            text={
                "left": env.vehicle.dist_to_left,
                "right": env.vehicle.dist_to_right,
                "white_lane_line": env.vehicle.on_white_continuous_line
            }
        )
        if d:
            print("Reset")
            env.reset()
    env.close()
