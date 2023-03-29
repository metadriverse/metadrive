import copy

import numpy as np

from metadrive.scenario.metadrive_type import MetaDriveType
from metadrive.utils.coordinates_shift import waymo_to_metadrive_heading, waymo_to_metadrive_vector


def parse_vehicle_state(
    object_dict, time_idx, coordinate_transform, check_last_state=False, sim_time_interval=0.1
):
    assert object_dict["type"] == MetaDriveType.VEHICLE

    states = object_dict["state"]
    ret = {k: v[time_idx] for k, v in states.items()}

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
    if coordinate_transform:
        ret["position"] = waymo_to_metadrive_vector(states["position"][time_idx])
        ret["velocity"] = waymo_to_metadrive_vector(states["velocity"][time_idx])
    else:
        ret["position"] = states["position"][time_idx]
        ret["velocity"] = states["velocity"][time_idx]
    ret["heading"] = waymo_to_metadrive_heading(states["heading"][time_idx], coordinate_transform=coordinate_transform)

    ret["length"] = states["size"][time_idx][0]
    ret["width"] = states["size"][time_idx][1]

    ret["valid"] = states["valid"][time_idx]
    if time_idx < len(states["position"]) - 1:
        angular_velocity = (states["heading"][time_idx + 1] - states["heading"][time_idx]) / sim_time_interval
        ret["angular_velocity"] = waymo_to_metadrive_heading(angular_velocity, coordinate_transform=coordinate_transform)
    else:
        ret["angular_velocity"] = 0
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
        trajectory = waymo_to_metadrive_vector(trajectory)

    return trajectory
