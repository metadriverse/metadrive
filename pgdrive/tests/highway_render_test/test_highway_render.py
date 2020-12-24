import time

from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.utils import setup_logger

setup_logger(True)


class TestEnv(PGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "environment_num": 1,
                "manual_control": True,
                "use_render": True,
                "use_image": False,
                "use_topdown": True,
            }
        )


if __name__ == "__main__":
    env = TestEnv()
    o = env.reset()
    s = time.time()
    for i in range(1, 100000):
        o, r, d, info = env.step(env.action_space.sample())
        env.render()
        if d:
            env.reset()
        if i % 1000 == 0:
            print("Steps: {}, Time: {}".format(i, time.time() - s))
    env.close()
