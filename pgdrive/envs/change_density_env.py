import logging

import numpy as np
from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.pg_config.pg_config import PgConfig
from pgdrive.utils import setup_logger

setup_logger(True)


class ChangeDensityEnv(PGDriveEnv):
    @staticmethod
    def default_config() -> PgConfig:
        config = PGDriveEnv.default_config()
        config.add("change_density", True)
        config.add("density_min", 0.0)
        config.add("density_max", 0.4)
        return config

    def __init__(self, config):
        super(ChangeDensityEnv, self).__init__(config)
        self.density_list = dict()
        for seed in self.maps.keys():
            self.density_list[seed] = np.random.uniform(self.config["density_min"], self.config["density_max"])

    def reset(self):
        if self.config["change_density"]:
            self.update_density()
        super(ChangeDensityEnv, self).reset()

    def update_density(self):
        assert self.config["change_density"]
        density = self.density_list[self.current_seed]
        self.config["traffic_density"] = density
        logging.debug("Environment opponent vehicle density is set to: {}".format(density))


if __name__ == '__main__':
    # Testing
    env = ChangeDensityEnv(config=dict(environment_num=10))
    for _ in range(10):
        env.reset()
    env.close()
