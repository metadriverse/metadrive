import gym
import numpy as np

from metadrive.obs.state_obs import LidarStateObservation
from metadrive.utils import clip


class NuPlanObservation(LidarStateObservation):
    MAX_LATERAL_DIST = 20

    def __init__(self, *args, **kwargs):
        super(NuPlanObservation, self).__init__(*args, **kwargs)
        self.lateral_dist = 0

    @property
    def observation_space(self):
        shape = list(self.state_obs.observation_space.shape)
        if self.config["lidar"]["num_lasers"] > 0 and self.config["lidar"]["distance"] > 0:
            # Number of lidar rays and distance should be positive!
            lidar_dim = self.config["lidar"]["num_lasers"] + self.config["lidar"]["num_others"] * 4
            if self.config["lidar"]["add_others_navi"]:
                lidar_dim += self.config["lidar"]["num_others"] * 4
            shape[0] += lidar_dim
        shape[0] += 1  # add one dim for sensing lateral distance to the sdc trajectory
        return gym.spaces.Box(-0.0, 1.0, shape=tuple(shape), dtype=np.float32)

    def state_observe(self, vehicle):
        ret = super(NuPlanObservation, self).state_observe(vehicle)
        lateral_obs = self.lateral_dist / self.MAX_LATERAL_DIST
        return np.concatenate([ret, [clip((lateral_obs + 1) / 2, 0.0, 1.0)]])

    def reset(self, env, vehicle=None):
        super(NuPlanObservation, self).reset(env, vehicle)
        self.lateral_dist = 0


class WaymoObservation(LidarStateObservation):
    MAX_LATERAL_DIST = 20

    def __init__(self, *args, **kwargs):
        super(WaymoObservation, self).__init__(*args, **kwargs)
        self.lateral_dist = 0

    @property
    def observation_space(self):
        shape = list(self.state_obs.observation_space.shape)
        if self.config["lidar"]["num_lasers"] > 0 and self.config["lidar"]["distance"] > 0:
            # Number of lidar rays and distance should be positive!
            lidar_dim = self.config["lidar"]["num_lasers"] + self.config["lidar"]["num_others"] * 4
            if self.config["lidar"]["add_others_navi"]:
                lidar_dim += self.config["lidar"]["num_others"] * 4
            shape[0] += lidar_dim
        shape[0] += 1  # add one dim for sensing lateral distance to the sdc trajectory
        return gym.spaces.Box(-0.0, 1.0, shape=tuple(shape), dtype=np.float32)

    def state_observe(self, vehicle):
        ret = super(WaymoObservation, self).state_observe(vehicle)
        lateral_obs = self.lateral_dist / self.MAX_LATERAL_DIST
        return np.concatenate([ret, [clip((lateral_obs + 1) / 2, 0.0, 1.0)]])

    def reset(self, env, vehicle=None):
        super(WaymoObservation, self).reset(env, vehicle)
        self.lateral_dist = 0
