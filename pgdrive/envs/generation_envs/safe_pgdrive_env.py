from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.pg_config import PGConfig


class SafePGDriveEnv(PGDriveEnv):
    def default_config(self) -> PGConfig:
        extra_config = {
            "accident_prob": 0.5,
            "crash_vehicle_cost": 1,
            "crash_object_cost": 1,
            "crash_vehicle_penalty": 0.,
            "crash_object_penalty": 0.,
            "out_of_road_cost": 0.,  # only give penalty for out_of_road
            "traffic_density": 0.2,
        }
        config = super(SafePGDriveEnv, self).default_config()
        config.extend_config_with_unknown_keys(extra_config)
        return config

    def custom_info_callback(self):
        self.step_info["cost"] = 0
        if self.step_info["crash_vehicle"]:
            self.step_info["cost"] = self.config["crash_vehicle_cost"]
            self.done = False
        elif self.step_info["crash_object"]:
            self.step_info["cost"] = self.config["crash_object_cost"]
            self.done = False
        elif self.step_info["out_of_road"]:
            self.step_info["cost"] = self.config["out_of_road_cost"]

    # def reward(self, action):
    #     """
    #     **No** lateral factor reward func
    #     """
    #     current_lane = self.vehicle.lane
    #     long_last, _ = current_lane.local_coordinates(self.vehicle.last_position)
    #     long_now, lateral_now = current_lane.local_coordinates(self.vehicle.position)
    #
    #     reward = 0.0
    #     if abs(lateral_now) <= self.current_map.lane_width / 2:
    #         # Out of road will get no reward
    #         reward += self.config["driving_reward"] * (long_now - long_last)
    #         reward += self.config["speed_reward"] * (self.vehicle.speed / self.vehicle.max_speed)
    #
    #     # Penalty for waiting
    #     if self.vehicle.speed < 1:
    #         reward -= self.config["low_speed_penalty"]  # encourage car
    #     reward -= self.config["general_penalty"]
    #
    #     self.step_info["raw_step_reward"] = reward
    #
    #     return reward


if __name__ == "__main__":
    env = SafePGDriveEnv(
        {
            "manual_control": True,
            "use_render": True,
            "environment_num": 100,
            "start_seed": 75,
            "debug": True,
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
        env.render(text={"cost": total_cost, "seed": env.current_map.random_seed})
        if d:
            total_cost = 0
            print("Reset")
            env.reset()
    env.close()
