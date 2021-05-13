from pgdrive.constants import TerminationState
from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.utils import PGConfig
from pgdrive.utils.math_utils import clip


class SafePGDriveEnv(PGDriveEnv):
    def default_config(self) -> PGConfig:
        config = super(SafePGDriveEnv, self).default_config()
        config.update(
            {
                "environment_num": 100,
                "accident_prob": 0.5,
                "safe_rl_env": True,  # Should always be True. But we just leave it here for historical reason.

                # ===== reward scheme =====
                "crash_vehicle_penalty": 0.,
                "crash_object_penalty": 0.,
                "out_of_road_penalty": 0.,

                # ===== cost scheme
                "crash_vehicle_cost": 1,
                "crash_object_cost": 0.5,
                "out_of_road_cost": 1.,  # only give penalty for out_of_road
                "traffic_density": 0.2,
                "use_lateral": False
            },
            allow_overwrite=True
        )
        return config

    def done_function(self, vehicle_id: str):
        done, done_info = super(SafePGDriveEnv, self).done_function(vehicle_id)
        if self.config["safe_rl_env"]:
            if done_info[TerminationState.CRASH_VEHICLE]:
                done = False
            elif done_info[TerminationState.CRASH_OBJECT]:
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
        reward = 0.0
        if self.config["use_lateral"]:
            lateral_factor = clip(
                1 - 2 * abs(lateral_now) / vehicle.routing_localization.get_current_lane_width(), 0.0, 1.0
            )
        else:
            lateral_factor = 1.0
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor

        # Penalty for waiting
        low_speed_penalty = 0
        if vehicle.speed < 1:
            low_speed_penalty = self.config["low_speed_penalty"]  # encourage car
        reward -= low_speed_penalty
        reward -= self.config["general_penalty"]

        reward += self.config["speed_reward"] * (vehicle.speed / vehicle.max_speed)
        step_info["step_reward"] = reward

        # for done
        if vehicle.crash_vehicle:
            reward = self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = self.config["crash_object_penalty"]
        elif vehicle.out_of_route:
            reward = self.config["out_of_road_penalty"]
        elif vehicle.crash_sidewalk:
            reward = self.config["out_of_road_penalty"]
        elif vehicle.arrive_destination:
            reward += self.config["success_reward"]

        return reward, step_info

    def cost_function(self, vehicle_id: str):
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()
        step_info["cost"] = 0
        if vehicle.crash_vehicle:
            step_info["cost"] = self.config["crash_vehicle_cost"]
        elif vehicle.crash_object:
            step_info["cost"] = self.config["crash_object_cost"]
        elif vehicle.out_of_route or vehicle.crash_sidewalk:
            step_info["cost"] = self.config["out_of_road_cost"]
        return step_info['cost'], step_info


if __name__ == "__main__":
    env = SafePGDriveEnv(
        {
            "accident_prob": 1.0,
            "manual_control": True,
            "use_render": True,
            "environment_num": 100,
            "start_seed": 139,
            "out_of_road_cost": 1,
            # "debug": True,
            "cull_scene": True,
            "pg_world_config": {
                "pstats": True
            },
            "vehicle_config": {
                "show_lidar": True,
                "show_side_detector": True,
                "show_lane_line_detector": True,
                "side_detector": dict(num_lasers=2, distance=50),  # laser num, distance
                "lane_line_detector": dict(num_lasers=2, distance=20),  # laser num, distance
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
            print("done_cost:{}".format(info["cost"]))
            print("Reset")
            env.reset()
    env.close()
