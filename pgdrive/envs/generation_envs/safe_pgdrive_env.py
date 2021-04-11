from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.utils import PGConfig


class SafePGDriveEnv(PGDriveEnv):
    def default_config(self) -> PGConfig:
        config = super(SafePGDriveEnv, self).default_config()
        config.update(
            {
                "accident_prob": 0.5,
                "crash_vehicle_cost": 1,
                "crash_object_cost": 1,
                "crash_vehicle_penalty": 0.,
                "crash_object_penalty": 0.,
                "out_of_road_cost": 0.,  # only give penalty for out_of_road
                "traffic_density": 0.2,
                "use_lateral": False
            },
            allow_overwrite=True
        )
        return config

    def done_function(self, vehicle_id: str):
        done, done_info = super(SafePGDriveEnv, self).done_function(vehicle_id)
        if done_info["crash_vehicle"]:
            done = False
        elif done_info["crash_object"]:
            done = False
        return done, done_info

    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()

        # Reward for moving forward in current lane
        current_lane = vehicle.lane if vehicle.lane in vehicle.routing_localization.current_ref_lanes else \
            vehicle.routing_localization.current_ref_lanes[0]
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral"]:
            lateral_factor = clip(
                1 - 2 * abs(lateral_now) / vehicle.routing_localization.get_current_lane_width(), 0.0, 1.0
            )
        else:
            lateral_factor = 1.0
        current_road = vehicle.current_road
        positive_road = 1 if not current_road.is_negative_road() else -1

        reward = 0.0
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor * positive_road
        reward += self.config["speed_reward"] * (vehicle.speed / vehicle.max_speed) * positive_road

        step_info["step_reward"] = reward

        if vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
            print(reward)
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
            print(reward)
        elif vehicle.out_of_route:
            reward = -self.config["out_of_road_penalty"]
            print(reward)
        elif vehicle.arrive_destination:
            reward = +self.config["success_reward"]
        return reward, step_info


if __name__ == "__main__":
    env = SafePGDriveEnv(
        {
            "manual_control": True,
            "use_render": True,
            "environment_num": 100,
            "start_seed": 75,
            # "debug": True,
            "cull_scene": True,
            "pg_world_config": {
                "pstats": True
            }
        }
    )

    o = env.reset()
    total_cost = 0
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        total_cost += info["cost"]
        env.render(text={"cost": total_cost, "seed": env.current_map.random_seed, "reward": r})
        if d:
            total_cost = 0
            print("Reset")
            env.reset()
    env.close()
