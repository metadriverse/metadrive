import gym
import numpy as np
from pgdrive.envs.pgdrive_env_v2 import PGDriveEnvV2
from pgdrive.obs.observation_type import LidarStateObservation, ObservationType
from pgdrive.utils import PGConfig


class MinimalObservation(LidarStateObservation):
    @property
    def observation_space(self):
        shape = list(self.state_obs.observation_space.shape)
        shape[0] += self.config["lidar"]["num_others"] * 4
        return gym.spaces.Box(-0.0, 1.0, shape=tuple(shape), dtype=np.float32)

    def observe(self, vehicle):
        state = self.state_obs.observe(vehicle)
        other_v_info = []
        assert self.config["lidar"]["num_others"] > 0
        other_v_info += vehicle.lidar.get_surrounding_vehicles_info(vehicle, self.config["lidar"]["num_others"])
        return np.concatenate((state, np.asarray(other_v_info)))


class PGDriveEnvV2Minimal(PGDriveEnvV2):
    @classmethod
    def default_config(cls) -> PGConfig:
        config = super(PGDriveEnvV2Minimal, cls).default_config()
        config["vehicle_config"]["lidar"]["num_others"] = 2
        config["vehicle_config"]["state_set"] = 0
        return config

    def get_single_observation(self, vehicle_config: "PGConfig") -> "ObservationType":
        return MinimalObservation(vehicle_config)


if __name__ == '__main__':

    def _act(env, action):
        assert env.action_space.contains(action)
        obs, reward, done, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isscalar(reward)
        assert isinstance(info, dict)

    env = PGDriveEnvV2Minimal()
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
