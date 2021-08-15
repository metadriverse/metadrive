from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.utils import setup_logger


class TestEnv(PGDriveEnv):
    def __init__(self):
        """
        TODO a small bug exists in scene 9 (30 blocks), traffic density > 0, respawn mode
        """
        super(TestEnv, self).__init__(
            {
                "environment_num": 10,
                "traffic_density": .2,
                "traffic_mode": "trigger",
                "start_seed": 12,
                # "_disable_detector_mask":True,
                "onscreen_message": True,
                # "debug_physics_world": True,
                "pstats": True,
                "global_light": True,
                # "debug_static_world":True,
                "cull_scene": False,
                "controller": "joystick",
                # "manual_control": True,
                "use_render": True,
                "decision_repeat": 5,
                "rgb_clip": True,
                "debug": True,
                "fast": True,
                # "map_config": {
                #     Map.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                #     Map.GENERATE_CONFIG: "SXO",
                #     Map.LANE_WIDTH: 3.5,
                #     Map.LANE_NUM: 3,
                # },
                "map": 8,
                "driving_reward": 1.0,
                "vehicle_config": {
                    "enable_reverse": True,
                    # "show_lidar": True,
                    # "spawn_lane_index":("1r1_0_", "1r1_1_", 0),
                    # "destination_node":"2R1_3_",
                    # "show_side_detector": True,
                    # "show_lane_line_detector": True,
                    # "side_detector": dict(num_lasers=2, distance=50),
                    # "lane_line_detector": dict(num_lasers=2, distance=50),
                    # # "show_line_to_dest": True,
                    # "show_dest_mark": True
                }
            }
        )


if __name__ == "__main__":
    setup_logger(True)
    env = TestEnv()
    import time
    start = time.time()
    o = env.reset()
    print(len(env.engine.traffic_manager._traffic_vehicles))
    for s in range(1, 100000):
        o, r, d, info = env.step([.0, 1.])
        # info["fuel"] = env.vehicle.energy_consumption
        env.render(text={"heading_diff": env.vehicle.heading_diff(env.vehicle.lane)})
        # assert env.observation_space.contains(o)
        # if (s + 1) % 100 == 0:
        #     print(
        #         "Finish {}/10000 simulation steps. Time elapse: {:.4f}. Average FPS: {:.4f}".format(
        #             s + 1,f
        #             time.time() - start, (s + 1) / (time.time() - start)
        #         )
        #     )
        # if s % 100 == 0:
        #     env.close()
        #     env.reset()
