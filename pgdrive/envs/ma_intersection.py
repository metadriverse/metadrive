from pgdrive.envs.pgdrive_env_v2 import PGDriveEnvV2
from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.scene_creator.vehicle.base_vehicle import BaseVehicle
from pgdrive.utils import setup_logger, PGConfig
from pgdrive.utils.pg_config import merge_dicts


class MultiAgentIntersectPGDrive(PGDriveEnvV2):
    # Currently only lane#0, #1 will be supported, #2 will be considered as out-of-road when spawning (hit side-lane)
    # TODO: Fix by either
    #  1. Change the side lane calculation to allow more car spawning space
    #  2. Limit the available slots to (lane_num - 1)

    # Max agents allowed will be (4 * 2) = 8, considering single intersection.MAX_SPOTS = 8
    DEFAULT_AGENT_NUM = 8
    agent_spawn_tracker = [
        {
            "value": (FirstBlock.NODE_1, FirstBlock.NODE_2, 0),
            "taken_agent": -1
        }, {
            "value": (FirstBlock.NODE_1, FirstBlock.NODE_2, 1),
            "taken_agent": -1
        }
    ]

    for i in range(3):
        for j in range(2):
            agent_spawn_tracker.append({"value": (f"-1X{i}_1_", f"-1X{i}_0_", j), "taken_agent": -1})

    def get_available_spot(self, agent_id):
        try:
            spot_obj = next(obj for obj in self.agent_spawn_tracker if obj["taken_agent"] == -1)
            spot_obj["taken_agent"] = agent_id
            print(f"Spot taken by agent{agent_id}")
            return spot_obj["value"]
        except StopIteration:
            print("Should be impossible since spot will be reopened if done/crash")

    """
    This serve as the base class for Multi-agent PGDrive!
    """

    @staticmethod
    def default_config() -> PGConfig:
        config = PGDriveEnvV2.default_config()
        target_vehicle_configs_dict = dict()
        for agent_id in range(MultiAgentIntersectPGDrive.DEFAULT_AGENT_NUM):
            target_vehicle_configs_dict[f"agent{agent_id}"] = dict()
        config.update(
            {
                "environment_num": 1,
                "traffic_density": 0.,
                "start_seed": 10,
                "vehicle_config": {
                    # "lane_line_detector": {
                    #     "num_lasers": 100
                    # }
                },
                "target_vehicle_configs": target_vehicle_configs_dict,
                "num_agents": MultiAgentIntersectPGDrive.DEFAULT_AGENT_NUM,
                "crash_done": True
            }
        )
        # Some collision bugs still exist, always set to False now!!!!
        # config.extend_config_with_unknown_keys({"crash_done": True})
        return config

    def __init__(self, config=None):
        config["target_vehicle_configs"] = dict()

        for agent_id in range(self.DEFAULT_AGENT_NUM):
            config["target_vehicle_configs"][f"agent{agent_id}"] = dict()
            config["target_vehicle_configs"][f"agent{agent_id}"]["born_longitude"] = 10
            config["target_vehicle_configs"][f"agent{agent_id}"]["born_lateral"] = 1
            config["target_vehicle_configs"][f"agent{agent_id}"]["born_lane_index"] = \
                self.get_available_spot(agent_id)
        super(MultiAgentIntersectPGDrive, self).__init__(config)

    def _process_extra_config(self, config) -> "PGConfig":
        ret_config = self.default_config().update(
            config, allow_overwrite=False, stop_recursive_update=["target_vehicle_configs"]
        )
        return ret_config

    def done_function(self, vehicle_id):
        vehicle = self.vehicles[vehicle_id]
        # crash will not done
        done, done_info = super(MultiAgentIntersectPGDrive, self).done_function(vehicle_id)
        if vehicle.crash_vehicle and not self.config["crash_done"]:
            done = False
            done_info["crash_vehicle"] = False
        elif vehicle.out_of_route and vehicle.on_lane and not vehicle.crash_sidewalk:
            done = False
            done_info["out_of_road"] = False
        return done, done_info

    def step(self, actions):
        actions = self._preprocess_marl_actions(actions)
        o, r, d, i = super(MultiAgentIntersectPGDrive, self).step(actions)
        self._after_vehicle_done(d)
        return o, r, d, i

    def reset(self, episode_data: dict = None):
        for idx in range(len(self.agent_spawn_tracker)):
            self.agent_spawn_tracker[idx]["taken_agent"] = -1
        for v in self.done_vehicles.values():
            v.chassis_np.node().setStatic(False)
        return super(MultiAgentIntersectPGDrive, self).reset(episode_data)

    def _preprocess_marl_actions(self, actions):
        # remove useless actions
        id_to_remove = []
        for id in actions.keys():
            if id in self.done_vehicles.keys():
                id_to_remove.append(id)
        for id in id_to_remove:
            actions.pop(id)
        return actions

    def _after_vehicle_done(self, dones: dict):
        for id, done in dones.items():
            if done and id in self.vehicles.keys():
                v = self.vehicles.pop(id)
                v.prepare_step([0, -1])
                self.done_vehicles[id] = v
        for v in self.done_vehicles.values():
            if v.speed < 1:
                v.chassis_np.node().setStatic(True)

    def _get_vehicles(self):
        return {
            name: BaseVehicle(self.pg_world, self._get_target_vehicle_config(new_config))
            for name, new_config in self.config["target_vehicle_configs"].items()
        }

    def _get_observations(self):
        return {
            name: self.get_single_observation(self._get_target_vehicle_config(new_config))
            for name, new_config in self.config["target_vehicle_configs"].items()
        }

    def _get_target_vehicle_config(self, extra_config: dict):
        """
        Newly introduce method
        """
        vehicle_config = merge_dicts(self.config["vehicle_config"], extra_config, allow_new_keys=False)
        return PGConfig(vehicle_config)


if __name__ == "__main__":
    setup_logger(True)

    env = MultiAgentIntersectPGDrive(
        {
            "map": "X",
            "use_render": True,
            "debug": False,
            "manual_control": True,
            "pg_world_config": {
                "pstats": False
            }
        }
    )

    o = env.reset()
    total_r = 0
    for i in range(1, 100000):
        o, r, d, info = env.step(env.action_space.sample())
        for r_ in r.values():
            total_r += r_
        # o, r, d, info = env.step([0,1])
        d.update({"total_r": total_r})
        env.render(text=d)
        if len(env.vehicles) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()
