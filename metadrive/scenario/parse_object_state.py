import copy

import numpy as np

from metadrive.type import MetaDriveType
from metadrive.utils.coordinates_shift import right_hand_to_left_hand_heading, right_hand_to_left_vector

from metadrive.utils.math_utils import compute_angular_velocity


def parse_vehicle_state(object_dict, time_idx, coordinate_transform, check_last_state=False, sim_time_interval=0.1):
    assert object_dict["type"] == MetaDriveType.VEHICLE

    states = object_dict["state"]

    epi_length = len(states["position"])
    if time_idx < 0:
        time_idx = epi_length + time_idx

    if time_idx >= len(states["position"]):
        time_idx = len(states["position"]) - 1
    if check_last_state:
        for current_idx in range(time_idx):
            p_1 = states["position"][current_idx][:2]
            p_2 = states["position"][current_idx + 1][:2]
            if np.linalg.norm(p_1 - p_2) > 100:
                time_idx = current_idx
                break

    ret = {k: v[time_idx] for k, v in states.items()}

    if coordinate_transform:
        ret["position"] = right_hand_to_left_vector(states["position"][time_idx][:2])
        ret["velocity"] = right_hand_to_left_vector(states["velocity"][time_idx])
    else:
        ret["position"] = states["position"][time_idx]
        ret["velocity"] = states["velocity"][time_idx]
    ret["heading_theta"] = right_hand_to_left_hand_heading(states["heading"][time_idx]
                                                           ) if coordinate_transform else states["heading"][time_idx]
    ret["heading"] = ret["heading_theta"]

    ret["length"] = states["size"][time_idx][0]
    ret["width"] = states["size"][time_idx][1]

    ret["valid"] = states["valid"][time_idx]
    if time_idx < len(states["position"]) - 1:
        angular_velocity = compute_angular_velocity(
            initial_heading=states["heading"][time_idx],
            final_heading=states["heading"][time_idx + 1],
            dt=sim_time_interval
        )
        ret["angular_velocity"] = right_hand_to_left_hand_heading(
            angular_velocity
        ) if coordinate_transform else angular_velocity
    else:
        ret["angular_velocity"] = 0

    # Retrieve vehicle type
    ret["vehicle_class"] = None
    if "spawn_info" in object_dict["metadata"]:
        type_module, type_cls_name = object_dict["metadata"]["spawn_info"]["type"]
        import importlib
        module = importlib.import_module(type_module)
        cls = getattr(module, type_cls_name)
        ret["vehicle_class"] = cls

    return ret


def parse_full_trajectory(object_dict, coordinate_transform):
    positions = object_dict["state"]["position"]
    index = len(positions)
    for current_idx in range(len(positions) - 1):
        p_1 = positions[current_idx][:2]
        p_2 = positions[current_idx + 1][:2]
        if np.linalg.norm(p_1 - p_2) > 100:
            index = current_idx
            break
    positions = positions[:index]
    trajectory = copy.deepcopy(positions[:, :2])
    # convert to metadrive coordinate
    if coordinate_transform:
        trajectory = right_hand_to_left_vector(trajectory)

    return trajectory
