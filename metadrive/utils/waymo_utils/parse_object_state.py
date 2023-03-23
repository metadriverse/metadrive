import copy

import numpy as np

from metadrive.utils.coordinates_shift import waymo_to_metadrive_heading, waymo_to_metadrive_vector


def parse_vehicle_state(states, time_idx, check_last_state=False):
    ret = {}
    if time_idx >= len(states["position"]):
        time_idx = len(states["position"]) - 1
    if check_last_state:
        for current_idx in range(time_idx):
            p_1 = states["position"][current_idx][:2]
            p_2 = states["position"][current_idx + 1][:2]
            if np.linalg.norm(p_1 - p_2) > 100:
                time_idx = current_idx
                break

    ret["position"] = waymo_to_metadrive_vector(states["position"][time_idx][:2])
    ret["length"] = states["size"][time_idx][0]
    ret["width"] = states["size"][time_idx][1]
    ret["heading"] = waymo_to_metadrive_heading(states["heading"][time_idx])
    ret["velocity"] = waymo_to_metadrive_vector(states["velocity"][time_idx])
    ret["valid"] = states["valid"][time_idx]
    return ret


def parse_full_trajectory(states):
    positions = states["position"]
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
    trajectory = waymo_to_metadrive_vector(trajectory)

    return trajectory
