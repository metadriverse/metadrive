import logging

from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.utils import PGConfig, get_np_random


class ChangeDensityEnv(PGDriveEnv):
    @staticmethod
    def default_config() -> PGConfig:
        config = PGDriveEnv.default_config()
        config.update({
            "change_density": True,
            "density_min": 0.0,
            "density_max": 0.4,
        })
        return config

    def __init__(self, config):
        super(ChangeDensityEnv, self).__init__(config)
        self.density_dict = dict()
        self._random_state = get_np_random(0)  # Use a fixed seed
        for seed in self.maps.keys():
            self.density_dict[seed] = self._random_state.uniform(self.config["density_min"], self.config["density_max"])

    def reset(self, *args, **kwargs):
        if self.config["change_density"]:
            self.update_density()
        return super(ChangeDensityEnv, self).reset(*args, **kwargs)

    def update_density(self):
        assert self.config["change_density"]
        assert self.current_seed in self.density_dict, self.density_dict
        density = self.density_dict[self.current_seed]
        self.config["traffic_density"] = density
        logging.debug("Environment opponent vehicle density is set to: {}".format(density))


if __name__ == '__main__':
    # Testing
    env = ChangeDensityEnv(config=dict(environment_num=100))
    for _ in range(1000):
        env.reset()
        for _ in range(10):
            env.step(env.action_space.sample())
    env.close()
