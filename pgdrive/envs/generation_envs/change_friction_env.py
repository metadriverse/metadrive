import numpy as np

from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.pg_config import PGConfig
from pgdrive.scene_creator.vehicle.base_vehicle import BaseVehicle
from pgdrive.utils import get_np_random


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

    def _change_friction(self):
        if self.config["change_friction"] and self.vehicle is not None:
            self.for_each_vehicle(lambda v: v.destroy(self.pg_world))
            del self.vehicles

            # We reset the friction here!
            parameter = self.parameter_list[self.current_seed]
            v_config = self.config["vehicle_config"]
            v_config["wheel_friction"] = parameter["wheel_friction"]

            self.vehicles = {
                agent_id: BaseVehicle(self.pg_world, v_config)
                for agent_id, v_config in self.config["target_vehicle_configs"].items()
            }

            self.init_track_vehicle()

    def reset(self, episode_data: dict = None):
        """
        Reset the env, scene can be restored and replayed by giving episode_data
        Reset the environment or load an episode from episode data to recover is
        :param episode_data: Feed the episode data to replay an episode
        :return: None
        """
        self.lazy_init()  # it only works the first time when reset() is called to avoid the error when render

        self.dones = {agent_id: False for agent_id in self.vehicles.keys()}
        self.episode_steps = 0

        # clear world and traffic manager
        self.pg_world.clear_world()

        # select_map
        self.update_map(episode_data)

        self._change_friction()

        # reset main vehicle
        self.for_each_vehicle(lambda v: v.reset(self.current_map))

        # generate new traffic according to the map
        self.scene_manager.reset(
            self.current_map,
            self.vehicles,
            self.config["traffic_density"],
            self.config["accident_prob"],
            episode_data=episode_data
        )

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
