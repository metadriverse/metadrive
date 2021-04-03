from pgdrive.envs.pgdrive_env_v2 import PGDriveEnvV2
from pgdrive.tests.test_env.test_pgdrive_env import _act


def test_pgdrive_env_rgb():
    env = PGDriveEnvV2(dict(use_image=True))
    try:
        obs = env.reset()
        assert env.observation_space.contains(obs)
        _act(env, env.action_space.sample())
        for x in [-1, 0, 1]:
            env.reset()
            for y in [-1, 0, 1]:
                _act(env, [x, y])
    finally:
        env.close()


if __name__ == '__main__':
    test_pgdrive_env_rgb()
