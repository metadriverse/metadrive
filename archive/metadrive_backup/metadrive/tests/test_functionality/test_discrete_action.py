import gymnasium as gym

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils import setup_logger

setup_logger(debug=True)


def _test_destroy(config):
    env = MetaDriveEnv(config)
    try:
        env.reset()
        for i in range(1, 20):
            env.step([1, 1])

        env.close()
        env.reset()
        env.close()
        env.close()
        env.reset()
        env.reset()
        env.close()

        # Again!
        env = MetaDriveEnv(config)
        env.reset()
        for i in range(1, 20):
            env.step([1, 1])
        env.reset()
        env.close()
    finally:
        env.close()


def test_discrete_action():
    # Close and reset
    env = MetaDriveEnv(
        dict(
            discrete_action=True,
            use_multi_discrete=False,
            discrete_steering_dim=3,
            discrete_throttle_dim=5,
            action_check=True
        )
    )
    try:
        env.reset()
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 15
        v = env.agent
        policy = env.engine.get_policy(v.name)
        assert policy.convert_to_continuous_action(0) == (-1, -1)
        assert policy.convert_to_continuous_action(1) == (0, -1)
        assert policy.convert_to_continuous_action(2) == (1, -1)
        assert policy.convert_to_continuous_action(7) == (0, 0)
        assert policy.convert_to_continuous_action(14) == (1, 1)

        for _ in range(20):
            o, r, tm, tc, i = env.step(env.action_space.sample())

    finally:
        env.close()


def test_multi_discrete_action():
    # Close and reset
    env = MetaDriveEnv(
        dict(
            discrete_action=True,
            use_multi_discrete=True,
            discrete_steering_dim=3,
            discrete_throttle_dim=5,
            action_check=True
        )
    )
    try:
        env.reset()
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        assert env.action_space.shape == (2, )
        assert all(env.action_space.nvec == (3, 5))
        v = env.agent
        policy = env.engine.get_policy(v.name)
        assert policy.convert_to_continuous_action([0, 0]) == (-1, -1)
        assert policy.convert_to_continuous_action([1, 0]) == (0, -1)
        assert policy.convert_to_continuous_action([2, 0]) == (1, -1)
        assert policy.convert_to_continuous_action([1, 2]) == (0, 0)
        assert policy.convert_to_continuous_action([2, 4]) == (1, 1)

        for _ in range(20):
            o, r, tm, tc, i = env.step(env.action_space.sample())

    finally:
        env.close()


if __name__ == "__main__":
    test_discrete_action()
    test_multi_discrete_action()
