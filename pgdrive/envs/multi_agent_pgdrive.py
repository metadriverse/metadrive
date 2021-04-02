from pgdrive.envs.pgdrive_env_v2 import PGDriveEnvV2
from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.scene_creator.vehicle.base_vehicle import BaseVehicle
from pgdrive.utils import setup_logger, PGConfig
from pgdrive.utils.pg_config import merge_dicts


class MultiAgentPGDrive(PGDriveEnvV2):
    """
    This serve as the base class for Multi-agent PGDrive!
    """
    @staticmethod
    def default_config() -> PGConfig:
        config = PGDriveEnvV2.default_config()
        config.update(
            {
                "environment_num": 1,
                "traffic_density": 0.,
                "start_seed": 10,
                "map": "yY",
                "vehicle_config": {
                    "use_lane_line_detector": True
                },
                "target_vehicle_configs": {
                    "agent0": {
                        "born_longitude": 10,
                        "born_lateral": 1.5,
                        "born_lane_index": (FirstBlock.NODE_1, FirstBlock.NODE_2, 1),
                        # "show_lidar": True
                        "show_side_detector": True
                    },
                    "agent1": {
                        "born_longitude": 10,
                        # "show_lidar": True,
                        "born_lateral": -1,
                        "born_lane_index": (FirstBlock.NODE_1, FirstBlock.NODE_2, 0),
                    },
                    "agent2": {
                        "born_longitude": 10,
                        "born_lane_index": (FirstBlock.NODE_1, FirstBlock.NODE_2, 2),
                        # "show_lidar": True,
                        "born_lateral": 1,
                    },
                    "agent3": {
                        "born_longitude": 10,
                        # "show_lidar": True,
                        "born_lateral": 2,
                        "born_lane_index": (FirstBlock.NODE_1, FirstBlock.NODE_2, 0),
                    }
                },
                "num_agents": 4,
                "crash_done": True
            }
        )
        # Some collision bugs still exist, always set to False now!!!!
        # config.extend_config_with_unknown_keys({"crash_done": True})
        return config

    def __init__(self, config=None):
        super(MultiAgentPGDrive, self).__init__(config)

    def _process_extra_config(self, config) -> "PGConfig":
        ret_config = self.default_config().update(
            config, allow_overwrite=False, stop_recursive_update=["target_vehicle_configs"]
        )
        return ret_config

    def done_function(self, vehicle_id):
        vehicle = self.vehicles[vehicle_id]
        # crash will not done
        done, done_info = super(MultiAgentPGDrive, self).done_function(vehicle_id)
        if vehicle.crash_vehicle and not self.config["crash_done"]:
            done = False
            done_info["crash_vehicle"] = False
        elif vehicle.out_of_route and vehicle.on_lane and not vehicle.crash_sidewalk:
            done = False
            done_info["out_of_road"] = False
        return done, done_info

    def step(self, actions):
        actions = self._preprocess_marl_actions(actions)
        o, r, d, i = super(MultiAgentPGDrive, self).step(actions)
        self._after_vehicle_done(d)
        return o, r, d, i

    def reset(self, episode_data: dict = None):
        for v in self.done_vehicles.values():
            v.chassis_np.node().setStatic(False)
        return super(MultiAgentPGDrive, self).reset(episode_data)

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
    env = MultiAgentPGDrive(
        {
            "use_render": True,
            "debug": False,
            "manual_control": True,
            "pg_world_config": {
                "pstats": False
            },
            "target_vehicle_configs": {
                "agent0": {
                    "born_longitude": 10,
                    "born_lateral": 1.5,
                    "born_lane_index": (FirstBlock.NODE_1, FirstBlock.NODE_2, 1),
                    # "show_lidar": True
                    "show_side_detector": True
                },
                "agent1": {
                    "born_longitude": 10,
                    # "show_lidar": True,
                    "born_lateral": -1,
                    "born_lane_index": (FirstBlock.NODE_1, FirstBlock.NODE_2, 0),
                },
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
