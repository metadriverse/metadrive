import numpy as np
from metadrive.constants import DEFAULT_AGENT
from metadrive.constants import TerminationState
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.state_obs import LidarStateObservation

info_keys = [
    "cost", "velocity", "steering", "acceleration", "step_reward", TerminationState.CRASH_VEHICLE,
    TerminationState.OUT_OF_ROAD, TerminationState.SUCCESS
]


def _act(env, action):
    action = np.asarray(action, dtype=env.action_space.dtype)
    assert env.action_space.contains(action)
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert np.isscalar(reward)
    assert isinstance(info, dict)
    for k in info_keys:
        assert k in info


def test_obs_noise():
    env = MetaDriveEnv({"vehicle_config": {"lidar": {"gaussian_noise": 1.0, "dropout_prob": 1.0}}})
    try:
        obs, _ = env.reset()
        obs_cls = env.observations[DEFAULT_AGENT]
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
    env = MetaDriveEnv({"vehicle_config": {"lidar": {"gaussian_noise": 0.0, "dropout_prob": 0.0}}})
    try:
        obs, _ = env.reset()
        obs_cls = env.observations[DEFAULT_AGENT]
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
