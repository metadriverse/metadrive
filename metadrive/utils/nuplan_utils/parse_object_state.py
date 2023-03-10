import numpy as np

from metadrive.utils.coordinates_shift import nuplan_to_metadrive_heading, nuplan_to_metadrive_vector


def parse_ego_vehicle_state(state, nuplan_center):
    center = nuplan_center
    ret = {}
    ret["position"] = nuplan_to_metadrive_vector([state.waypoint.x, state.waypoint.y], center)
    ret["heading"] = nuplan_to_metadrive_heading(np.rad2deg(state.waypoint.heading))
    ret["velocity"] = nuplan_to_metadrive_vector([state.agent.velocity.x, state.agent.velocity.y])
    ret["valid"] = True
    ret["length"] = state.agent.box.length
    ret["width"] = state.agent.box.width
    return ret


def parse_ego_vehicle_trajectory(states, nuplan_center):
    traj = []
    for state in states:
        traj.append([state.waypoint.x, state.waypoint.y])
    trajectory = nuplan_to_metadrive_vector(traj, nuplan_center=nuplan_center)
    return trajectory


def parse_object_state(obj_state, nuplan_center):
    ret = {}
    ret["position"] = nuplan_to_metadrive_vector([obj_state.center.x, obj_state.center.y], nuplan_center)
    ret["heading"] = nuplan_to_metadrive_heading(obj_state.center.heading)
    ret["velocity"] = nuplan_to_metadrive_vector([obj_state.velocity.x, obj_state.velocity.y])
    ret["valid"] = True
    ret["length"] = obj_state.box.length
    ret["width"] = obj_state.box.width
    return ret
