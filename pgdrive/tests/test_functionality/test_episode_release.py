from pgdrive.envs.safe_pgdrive_env import SafePGDriveEnv
from pgdrive.utils import setup_logger


class TestEnv(SafePGDriveEnv):
    def __init__(self):
        super(TestEnv, self).__init__(
            {
                "use_render": False,
                "environment_num": 100,
                "accident_prob": .8,
                "traffic_density": 0.5,
                "debug": True
            }
        )


def test_episode_release():
    try:
        env = TestEnv()
        o = env.reset()
        for i in range(1, 10):
            env.step([1.0, 1.0])
            env.step([1.0, 1.0])
            env.step([1.0, 1.0])
            env.step([1.0, 1.0])
            env.reset()
            env.reset()
    finally:
        env.close()
