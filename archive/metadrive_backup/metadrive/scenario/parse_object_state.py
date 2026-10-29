import copy
import warnings

from metadrive.component.lane.point_lane import PointLane
from metadrive.utils.math import compute_angular_velocity
from metadrive.utils.math import norm


def get_max_valid_indicis(track, current_index):
    """
    Find the invalid timestep and get the trajectory before that step
    """
    states = track["state"]
    assert states["valid"][current_index], "Current index should be valid"
    end = len(states["valid"])
    for i, valid in enumerate(states["valid"][current_index + 1:], current_index + 1):
        if not valid:
            end = i
            break
    return current_index, end


def get_idm_route(traj_points, width=2):
    traj = PointLane(traj_points, width)
    return traj


def parse_object_state(object_dict, time_idx, check_last_state=False, sim_time_interval=0.1, include_z_position=False):
    """
    Parse object state of one time step
    """
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
            if norm(p_1[0] - p_2[0], p_1[1] - p_2[1]) > 100:
                time_idx = current_idx
                break

    ret = {k: v[time_idx] for k, v in states.items()}

    if include_z_position:
        ret["position"] = states["position"][time_idx]
    else:
        ret["position"] = states["position"][time_idx, :2]

    ret["velocity"] = states["velocity"][time_idx]

    ret["heading_theta"] = states["heading"][time_idx]

    ret["heading"] = ret["heading_theta"]

    # optional keys with scalar value:
    for k in ["length", "width", "height"]:
        if k in states:
            ret[k] = float(states[k][time_idx].item())

    ret["valid"] = states["valid"][time_idx]
    if time_idx < len(states["position"]) - 1 and states["valid"][time_idx] and states["valid"][time_idx + 1]:
        angular_velocity = compute_angular_velocity(
            initial_heading=states["heading"][time_idx],
            final_heading=states["heading"][time_idx + 1],
            dt=sim_time_interval
        )
        ret["angular_velocity"] = angular_velocity
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


def parse_full_trajectory(object_dict):
    """
    Parse object states for a whole trajectory
    """
    positions = object_dict["state"]["position"]
    index = len(positions)
    for current_idx in range(len(positions) - 1):
        p_1 = positions[current_idx][:2]
        p_2 = positions[current_idx + 1][:2]
        if norm(p_1[0] - p_2[0], p_1[1] - p_2[1]) > 100:
            index = current_idx
            break
    positions = positions[:index]
    trajectory = copy.deepcopy(positions[:, :2])

    return trajectory
