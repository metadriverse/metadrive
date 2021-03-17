from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.pg_config import PGConfig
from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.utils import setup_logger

setup_logger(True)


class MultiAgentPGDrive(PGDriveEnv):
    @staticmethod
    def default_config() -> PGConfig:
        config = PGDriveEnv.default_config()
        config.update(
            {
                "environment_num": 1,
                "traffic_density": 0.,
                "start_seed": 5,
                "map": "TrOCXR",
                "target_vehicle_configs": {
                    "agent0": {
                        "born_longitude": 10,
                        "born_lane_index": (FirstBlock.NODE_1, FirstBlock.NODE_2, 2),
                        "show_lidar": True
                    },
                    "agent1": {
                        "born_longitude": 10,
                        "show_lidar": True
                    }
                },
                "num_agents": 2,
            }
        )
        # Some collision bugs still exist, always set to False now!!!!
        config.extend_config_with_unknown_keys({"crash_done": False})
        return config

    def __init__(self, config=None):
        super(MultiAgentPGDrive, self).__init__(config)

    def done_function(self, vehicle):
        # crash will not done
        done, done_info = super(MultiAgentPGDrive, self).done_function(vehicle)
        if vehicle.crash_vehicle and not self.config["crash_done"]:
            done = False
            done_info["crash_vehicle"] = False
        return done, done_info

    def step(self, actions):
        # remove useless actions
        id_to_remove = []
        for id in actions.keys():
            if id in self.done_vehicles.keys():
                id_to_remove.append(id)
        for id in id_to_remove:
            actions.pop(id)

        o, r, d, i = super(MultiAgentPGDrive, self).step(actions)
        for id, done in d.items():
            if done and id in self.vehicles.keys():
                v = self.vehicles.pop(id)
                v.prepare_step([0, -1])
                self.done_vehicles[id] = v
        return o, r, d, i


if __name__ == "__main__":
    env = MultiAgentPGDrive({"use_render": True, "manual_control": True})
    o = env.reset()
    for i in range(1, 100000):
        o, r, d, info = env.step({"agent0": [-1, 0], "agent1": [0, 0]})
        # o, r, d, info = env.step([0,1])
        env.render(text=d)
        if len(env.vehicles) == 0:
            print("Reset")
            env.reset()
    env.close()
