from pgdrive.utils import clip

pg_reward_scheme = dict(
    # ===== Reward Scheme =====
    success_reward=20,
    out_of_road_penalty=5,
    crash_vehicle_penalty=10,
    crash_object_penalty=2,
    acceleration_penalty=0.0,
    steering_penalty=0.1,
    low_speed_penalty=0.0,
    driving_reward=1.0,
    general_penalty=0.0,
    speed_reward=0.1
)


def pg_reward_function(vehicle) -> float:
    """
    Override this func to get a new reward function
    :param vehicle: BaseVehicle
    :return: reward
    """
    action = vehicle.last_current_action[1]
    # Reward for moving forward in current lane
    current_lane = vehicle.lane
    long_last, _ = current_lane.local_coordinates(vehicle.last_position)
    long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

    # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
    reward = 0.0
    lateral_factor = clip(1 - 2 * abs(lateral_now) / vehicle.routing_localization.map.lane_width, 0.0, 1.0)
    reward += vehicle.vehicle_config["driving_reward"] * (long_now - long_last) * lateral_factor

    # Penalty for frequent steering
    steering_change = abs(vehicle.last_current_action[0][0] - vehicle.last_current_action[1][0])
    steering_penalty = vehicle.vehicle_config["steering_penalty"] * steering_change * vehicle.speed / 20
    reward -= steering_penalty

    # Penalty for frequent acceleration / brake
    acceleration_penalty = vehicle.vehicle_config["acceleration_penalty"] * ((action[1])**2)
    reward -= acceleration_penalty

    # Penalty for waiting
    low_speed_penalty = 0
    if vehicle.speed < 1:
        low_speed_penalty = vehicle.vehicle_config["low_speed_penalty"]  # encourage car
    reward -= low_speed_penalty
    reward -= vehicle.vehicle_config["general_penalty"]

    reward += vehicle.vehicle_config["speed_reward"] * (vehicle.speed / vehicle.max_speed)
    vehicle.step_info["step_reward"] = reward

    # for done
    if vehicle.step_info["crash_vehicle"]:
        reward -= vehicle.vehicle_config["crash_vehicle_penalty"]
    elif vehicle.step_info["crash_object"]:
        reward -= vehicle.vehicle_config["crash_object_penalty"]
    elif vehicle.step_info["out_of_road"]:
        reward -= vehicle.vehicle_config["out_of_road_penalty"]
    elif vehicle.step_info["arrive_dest"]:
        reward += vehicle.vehicle_config["success_reward"]

    return reward
