from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.utils import setup_logger

setup_logger(True)


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 1,
                "manual_control": True,
                "use_render": False,
                "use_image": False,
                "use_topdown": True,
            }
        )


if __name__ == "__main__":
    env = TestEnv()

    o = env.reset()
    for i in range(1, 100000):
        o, r, d, info = env.step([0.01, 0.1])
        env.render()
        # if d:
        #     print("Reset")
        #     env.reset()
    env.close()
