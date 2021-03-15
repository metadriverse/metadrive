import numpy as np
from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.pg_config import PGConfig
from pgdrive.scene_creator.vehicle.base_vehicle import BaseVehicle
from pgdrive.utils import get_np_random

from pgdrive.world.chase_camera import ChaseCamera


class ChangeFrictionEnv(PGDriveEnv):
    @staticmethod
    def default_config() -> PGConfig:
        config = PGDriveEnv.default_config()
        config.add("change_friction", True)
        config.add("friction_min", 0.8)
        config.add("friction_max", 1.2)
        config.update({"vehicle_config": {"wheel_friction": 1.0}})
        return config

    def __init__(self, config=None):
        super(ChangeFrictionEnv, self).__init__(config)
        self.parameter_list = dict()
        self._random_state = get_np_random(0)  # Use a fixed seed.
        for k in self.maps:
            self.parameter_list[k] = dict(
                wheel_friction=self._random_state.uniform(self.config["friction_min"], self.config["friction_max"])
            )

    def reset(self, episode_data: dict = None):
        """
        Reset the env, scene can be restored and replayed by giving episode_data
        Reset the environment or load an episode from episode data to recover is
        :param episode_data: Feed the episode data to replay an episode
        :return: None
        """
        self.lazy_init()  # it only works the first time when reset() is called to avoid the error when render

        self.dones = {a: False for a in self.multi_agent_action_space.keys()}
        self.takeover = False

        # clear world and traffic manager
        self.pg_world.clear_world()

        # select_map
        self.update_map(episode_data)

        # ===== Key difference! =====
        if self.config["change_friction"] and self.vehicle is not None:
            for v in self.vehicles.values():
                v.destroy(self.pg_world.physics_world)
            del self.vehicles
            # We reset the friction here!
            parameter = self.parameter_list[self.current_seed]
            v_config = self.config["vehicle_config"]
            v_config["wheel_friction"] = parameter["wheel_friction"]
            self.vehicles = {a: BaseVehicle(self.pg_world, v_config) for a in self.multi_agent_action_space.keys()}
            # for manual_control and main camera type
            if (self.config["use_render"] or self.config["use_image"]) and self.config["use_chase_camera"]:
                self.main_camera = ChaseCamera(
                    self.pg_world.cam, self.vehicle, self.config["camera_height"], 7, self.pg_world
                )
            for v in self.vehicles.values():
                self.add_modules_for_vehicle(v)

        # reset main vehicle
        # self.vehicle.reset(self.current_map, self.vehicle.born_place, 0.0)
        for v in self.vehicles.values():
            v.reset(self.current_map, v.born_place, 0.0)

        # generate new traffic according to the map
        self.scene_manager.reset(
            self.current_map,
            self.vehicles,
            self.config["traffic_density"],
            self.config["accident_prob"],
            episode_data=episode_data
        )

        self.front_vehicles = set()
        self.back_vehicles = set()
        return self._get_reset_return()


if __name__ == '__main__':
    env = ChangeFrictionEnv(config={"environment_num": 100, "start_seed": 1000, "change_friction": True})
    env.seed(100000)
    obs = env.reset()
    for s in range(100):
        action = np.array([0.0, 1.0])
        o, r, d, i = env.step(action)
        if d:
            env.reset()
