import numpy as np

from pgdrive.envs.pgdrive_env_v2 import PGDriveEnvV2
from pgdrive.obs.state_obs import LidarStateObservation
from pgdrive.tests.test_env.test_pgdrive_env import _act


def test_obs_noise():
    env = PGDriveEnvV2({"vehicle_config": {"lidar": {"gaussian_noise": 1.0, "dropout_prob": 1.0}}})
    try:
        obs = env.reset()
        obs_cls = env.observations[env.DEFAULT_AGENT]
        assert isinstance(obs_cls, LidarStateObservation)
        ret = obs_cls._add_noise_to_cloud_points([0.5, 0.5, 0.5], gaussian_noise=1.0, dropout_prob=1.0)
        np.testing.assert_almost_equal(np.array(ret), 0.0)
        assert env.observation_space.contains(obs)
        _act(env, env.action_space.sample())
        for x in [-1, 0, 1]:
            env.reset()
            for y in [-1, 0, 1]:
                _act(env, [x, y])
    finally:
        env.close()
    env = PGDriveEnvV2({"vehicle_config": {"lidar": {"gaussian_noise": 0.0, "dropout_prob": 0.0}}})
    try:
        obs = env.reset()
        obs_cls = env.observations[env.DEFAULT_AGENT]
        assert isinstance(obs_cls, LidarStateObservation)
        ret = obs_cls._add_noise_to_cloud_points([0.5, 0.5, 0.5], gaussian_noise=0.0, dropout_prob=0.0)
        assert not np.all(np.array(ret) == 0.0)
        np.testing.assert_equal(np.array(ret), np.array([0.5, 0.5, 0.5]))
        assert env.observation_space.contains(obs)
        _act(env, env.action_space.sample())
        for x in [-1, 0, 1]:
            env.reset()
            for y in [-1, 0, 1]:
                _act(env, [x, y])
    finally:
        env.close()


if __name__ == '__main__':
    test_obs_noise()
