from pgdrive.envs.safe_pgdrive_env import SafePGDriveEnv
from pgdrive.utils import setup_logger

if __name__ == "__main__":
    setup_logger(True)
    env = SafePGDriveEnv(
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

    o = env.reset()
    print("vehicle num", len(env.engine.traffic_manager.vehicles))
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        env.render(text={})
    env.close()
