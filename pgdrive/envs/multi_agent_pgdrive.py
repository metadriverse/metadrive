from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.utils import PGConfig, setup_logger

setup_logger(True)


class MultiAgentPGDrive(PGDriveEnv):
    @staticmethod
    def default_config() -> PGConfig:
        config = PGDriveEnv.default_config()
        config.update(
            {
                "environment_num": 1,
                "traffic_density": 0.,
                "start_seed": 10,
                "map": "yY",
                "target_vehicle_configs": {
                    "agent0": {
                        "born_longitude": 10,
                        "born_lateral": 1.5,
                        "born_lane_index": (FirstBlock.NODE_1, FirstBlock.NODE_2, 1),
                        # "show_lidar": True
                    },
                    "agent1": {
                        "born_longitude": 10,
                        # "show_lidar": True,
                        "born_lateral": -1,
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
                    }
                },
                "num_agents": 4,
            }
        )
        # Some collision bugs still exist, always set to False now!!!!
        config.extend_config_with_unknown_keys({"crash_done": True})
        return config

    def __init__(self, config=None):
        super(MultiAgentPGDrive, self).__init__(config)

    def done_function(self, vehicle):
        # crash will not done
        done, done_info = super(MultiAgentPGDrive, self).done_function(vehicle)
        if vehicle.crash_vehicle and not self.config["crash_done"]:
            done = False
            done_info["crash_vehicle"] = False
        elif vehicle.out_of_route and vehicle.on_lane and not vehicle.crash_sidewalk:
            done = False
            done_info["out_of_road"] = False
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

    def reward_function(self, vehicle):
        """
           Override this func to get a new reward function
           :param vehicle: BaseVehicle
           :return: reward
           """
        step_info = dict()

        # Reward for moving forward in current lane
        current_lane = vehicle.lane
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        reward = 0.0

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane

        lateral_factor = 1.0

        reward += vehicle.vehicle_config["driving_reward"] * (long_now - long_last) * lateral_factor

        reward += vehicle.vehicle_config["speed_reward"] * (vehicle.speed / vehicle.max_speed)
        step_info["step_reward"] = reward

        if vehicle.crash_vehicle:
            reward = -vehicle.vehicle_config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -vehicle.vehicle_config["crash_object_penalty"]
        elif vehicle.arrive_destination:
            reward = +vehicle.vehicle_config["success_reward"]

        return reward, step_info


if __name__ == "__main__":
    env = MultiAgentPGDrive(
        {
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
        o, r, d, info = env.step({"agent0": [-1, 0], "agent1": [0, 0], "agent2": [-1, 0], "agent3": [0, 0]})
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
