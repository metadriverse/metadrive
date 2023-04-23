from metadrive.utils.coordinates_shift import nuplan_to_metadrive_heading, nuplan_to_metadrive_vector
from metadrive.utils.math import compute_angular_velocity


def parse_ego_vehicle_state(state, nuplan_center):
    center = nuplan_center
    ret = {}
    ret["position"] = nuplan_to_metadrive_vector([state.waypoint.x, state.waypoint.y], center)
    ret["heading"] = nuplan_to_metadrive_heading(state.waypoint.heading)
    ret["velocity"] = nuplan_to_metadrive_vector([state.agent.velocity.x, state.agent.velocity.y])
    ret["angular_velocity"] = nuplan_to_metadrive_heading(state.dynamic_car_state.angular_velocity)
    ret["valid"] = True
    ret["length"] = state.agent.box.length
    ret["width"] = state.agent.box.width
    return ret


def parse_ego_vehicle_state_trajectory(scenario, nuplan_center):
    data = [
        parse_ego_vehicle_state(scenario.get_ego_state_at_iteration(i), nuplan_center)
        for i in range(scenario.get_number_of_iterations())
    ]
    for i in range(len(data) - 1):
        data[i]["angular_velocity"] = compute_angular_velocity(
            initial_heading=data[i]["heading"], final_heading=data[i + 1]["heading"], dt=scenario.database_interval
        )
    return data


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
