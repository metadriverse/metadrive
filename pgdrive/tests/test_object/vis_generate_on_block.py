from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.utils import setup_logger

setup_logger(True)


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 1,
                # "traffic_density": 0.,
                "traffic_mode": "hybrid",
                "start_seed": 3,
                "use_render": True,
                "accident_prob": 1.0,
                # "controller":"joystick",
                "manual_control": True,
                "map": 7
            }
        )


if __name__ == "__main__":
    env = TestEnv()

    o = env.reset()
    print("vehicle num", len(env.scene_manager.traffic_mgr.vehicles))
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        env.render(text={})
        # if d:
        #     print("Reset")
        #     env.reset()
    env.close()
