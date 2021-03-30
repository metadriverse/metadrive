from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.utils import setup_logger


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 1,
                "traffic_density": 0.,
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
                # "debug":True,
                "map": "XTX",
                "target_vehicle_configs": {
                    "agent0": {
                        "born_longitude": 40,
                        "show_lidar": True,
                    },
                    "agent1": {
                        "born_longitude": 10,
                        "show_lidar": True,
                    }
                },
                "num_agents": 2
            }
        )


if __name__ == "__main__":
    setup_logger(True)
    env = TestEnv()

    o = env.reset()
    print("vehicle num", len(env.scene_manager.traffic_mgr.vehicles))
    for i in range(1, 100000):
        o, r, d, info = env.step({"agent0": [0, 0], "agent1": [0, 0]})
        # o, r, d, info = env.step([0,1])
        env.render(text=d)
        if True in d.values():
            print("Reset")
            env.reset()
    env.close()
