from pgdrive.envs.safe_pgdrive_env import SafePGDriveEnv
from pgdrive.utils import setup_logger


class TestEnv(SafePGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "use_render": True,
                "manual_control": True,
                "environment_num": 100,
                "accident_prob": 1.0,
                "vehicle_config": {
                    "show_lidar": True
                }
            }
        )


if __name__ == "__main__":
    setup_logger(True)
    env = TestEnv()

    o = env.reset()
    print("vehicle num", len(env.engine.traffic_manager.vehicles))
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        env.render(
            text={
                "vehicle_num": len(env.engine.traffic_manager.traffic_vehicles),
                "dist_to_left:": env.vehicle.dist_to_left_side,
                "dist_to_right:": env.vehicle.dist_to_right_side
            }
        )
    env.close()
