import numpy as np

from metadrive.utils.coordinates_shift import nuplan_2_metadrive_heading, nuplan_2_metadrive_position


def parse_ego_vehicle_state(state, nuplan_center):
    center = nuplan_center
    ret = {}
    ret["position"] = nuplan_2_metadrive_position([state.waypoint.x, state.waypoint.y], center)
    ret["heading"] = nuplan_2_metadrive_heading(np.rad2deg(state.waypoint.heading))
    ret["velocity"] = [state.agent.velocity.x, state.agent.velocity.y]
    ret["valid"] = True
    return ret


def parse_ego_vehicle_trajectory(states, nuplan_center):
    center = nuplan_center
    traj = []
    for state in states:
        traj.append([state.waypoint.x - center[0], state.waypoint.y - center[1]])

    trajectory = np.array(traj)
    trajectory *= [1, -1]
    return trajectory
