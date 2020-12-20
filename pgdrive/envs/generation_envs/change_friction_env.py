import logging

import numpy as np
from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.pg_config import PgConfig
from pgdrive.scene_creator.ego_vehicle.base_vehicle import BaseVehicle
from pgdrive.utils import get_np_random
from pgdrive.world.chase_camera import ChaseCamera


class ChangeFrictionEnv(PGDriveEnv):
    @staticmethod
    def default_config() -> PgConfig:
        config = PGDriveEnv.default_config()
        config.add("change_friction", True)
        config.add("friction_min", 0.6)
        config.add("friction_max", 1.2)
        return config

    def __init__(self, config=None):
        super(ChangeFrictionEnv, self).__init__(config)
        self.parameter_list = dict()
        self._random_state = get_np_random(0)  # Use a fixed seed.
        for k in self.maps:
            self.parameter_list[k] = dict(
                wheel_friction=self._random_state.uniform(self.config["friction_min"], self.config["friction_max"])
            )

    def reset(self):
        self.lazy_init()  # it only works the first time when reset() is called to avoid the error when render
        self.done = False

        # clear world and traffic manager
        self.pg_world.clear_world()
        # select_map
        self.select_map()

        if self.config["change_friction"] and self.vehicle is not None:
            self.vehicle.destroy(self.pg_world.physics_world)
            del self.vehicle
            self.main_camera = None

            parameter = self.parameter_list[self.current_seed]
            v_config = self.config["vehicle_config"]
            v_config["wheel_friction"] = parameter["wheel_friction"]
            self.vehicle = BaseVehicle(self.pg_world, v_config)

            # for manual_control and main camera type
            if (self.config["use_render"] or self.config["use_image"]) and self.config["use_chase_camera"]:
                self.main_camera = ChaseCamera(
                    self.pg_world.cam, self.vehicle, self.config["camera_height"], 7, self.pg_world
                )
            # add sensors
            self.add_modules_for_vehicle()

            logging.debug("The friction is changed to: ", parameter["wheel_friction"])

        # reset main vehicle
        self.vehicle.reset(self.current_map, self.vehicle.born_place, 0.0)

        # generate new traffic according to the map
        assert self.vehicle is not None
        self.traffic_manager.generate_traffic(
            self.pg_world, self.current_map, self.vehicle, self.config["traffic_density"]
        )
        o = self._get_reset_return()
        return o


if __name__ == '__main__':
    env = ChangeFrictionEnv(config={"environment_num": 100, "start_seed": 1000})
    env.seed(100000)
    obs = env.reset()
    for s in range(10000):
        action = np.array([0.0, 1.0])
        o, r, d, i = env.step(action)
        if d:
            env.reset()
