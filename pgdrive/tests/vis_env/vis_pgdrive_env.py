from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.utils import setup_logger

if __name__ == "__main__":
    setup_logger(True)
    env = PGDriveEnv(
        {
            "environment_num": 1,
            "traffic_density": .2,
            "traffic_mode": "trigger",
            "start_seed": 12,
            # "_disable_detector_mask":True,
            "onscreen_message": True,
            # "debug_physics_world": True,
            "global_light": True,
            # "debug_static_world":True,
            "cull_scene": False,
            # "controller": "joystick",
            # "manual_control": True,
            "use_render": True,
            "decision_repeat": 5,
            "rgb_clip": True,
            # "debug": True,
            "random_lane_num": True,
            "fast": True,

            # "map_config": {
            #     Map.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
            #     Map.GENERATE_CONFIG: "OXO",
            #     Map.LANE_WIDTH: 3.5,
            #     Map.LANE_NUM: 3,
            # },
            "pstats": True,
            "discrete_action": True,
            "map": "SSS",
            "random_traffic": False,
            "random_lane_width": True,
            "random_agent_model": True,
            "load_map_from_json": False,
            "driving_reward": 1.0,
            "vehicle_config": {
                "enable_reverse": True,
                # "random_color": True
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
    import time

    start = time.time()
    o = env.reset()

    for s in range(1, 100000):
        o, r, d, info = env.step([2, 4])
        # info["fuel"] = env.vehicle.energy_consumption
        env.render(
            text={
                "heading_diff": env.vehicle.heading_diff(env.vehicle.lane),
                "engine_force": env.vehicle.config["max_engine_force"],
                "current_seed": env.current_seed,
                "lane_width": env.vehicle.lane.width
            }
        )
        # assert env.observation_space.contains(o)
        # if (s + 1) % 100 == 0:
        #     print(
        #         "Finish {}/10000 simulation steps. Time elapse: {:.4f}. Average FPS: {:.4f}".format(
        #             s + 1,f
        #             time.time() - start, (s + 1) / (time.time() - start)
        #         )
        #     )
        # if d:
        #     # env.close()
        #     print(len(env.engine._spawned_objects))
        #     env.reset()
