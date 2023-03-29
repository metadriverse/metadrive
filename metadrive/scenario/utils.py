import copy
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
from metadrive.component.traffic_participants.cyclist import Cyclist
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import DATA_VERSION, DEFAULT_AGENT
from metadrive.scenario import MetaDriveType, ScenarioDescription as SD


def draw_map(map_features, show=False):
    figure(figsize=(8, 6), dpi=500)
    for key, value in map_features.items():
        if value.get("type", None) == MetaDriveType.LANE_CENTER_LINE:
            plt.scatter([x[0] for x in value["polyline"]], [y[1] for y in value["polyline"]], s=0.1)
        elif value.get("type", None) == "road_edge":
            plt.scatter([x[0] for x in value["polyline"]], [y[1] for y in value["polyline"]], s=0.1, c=(0, 0, 0))
        # elif value.get("type", None) == "road_line":
        #     plt.scatter([x[0] for x in value["polyline"]], [y[1] for y in value["polyline"]], s=0.5, c=(0.8,0.8,0.8))
    if show:
        plt.show()


def get_type_from_class(obj_class):
    if issubclass(obj_class, BaseVehicle) or obj_class is BaseVehicle:
        return MetaDriveType.VEHICLE
    elif issubclass(obj_class, Pedestrian) or obj_class is Pedestrian:
        return MetaDriveType.PEDESTRIAN
    elif issubclass(obj_class, Cyclist) or obj_class is Cyclist:
        return MetaDriveType.CYCLIST
    else:
        return MetaDriveType.OTHER


def convert_recorded_scenario_exported(record_episode, scenario_log_interval=0.1):
    """
    This function utilizes the recorded data natively emerging from MetaDrive run.
    The output data structure follows MetaDrive data format, but some changes might happen compared to original data.
    For example, MetaDrive InterpolateLane will reformat the Lane data and making all waypoints equal distancing.
    We call this lane sampling rate, which is 0.2m in MetaDrive but might different in other dataset.
    """
    result = SD()

    result[SD.ID] = "{}-{}".format(record_episode["map_data"]["map_type"], record_episode["scenario_index"])

    result[SD.VERSION] = DATA_VERSION

    result["map_features"] = record_episode["map_data"]["map_features"]

    scenario_log_interval = scenario_log_interval or record_episode["global_config"]["physics_world_step_size"]

    frames_skip = int(scenario_log_interval / record_episode["global_config"]["physics_world_step_size"])

    frames = [step_frame_list[0] for step_frame_list in record_episode["frame"]]

    episode_len = len(frames)
    result[SD.LENGTH] = episode_len

    result[SD.METADATA] = {}
    result[SD.METADATA][SD.METADRIVE_PROCESSED] = True
    result[SD.METADATA]["dataset"] = "metadrive"
    result[SD.METADATA]["seed"] = record_episode["global_seed"]
    result[SD.METADATA]["scenario_id"] = record_episode["scenario_index"]
    result[SD.METADATA][SD.CREATED_TIME] = time.time()
    result[SD.METADATA][SD.COORDINATE] = MetaDriveType.COORDINATE_METADRIVE
    result[SD.METADATA][SD.SDC_ID] = str(frames[0]._agent_to_object[DEFAULT_AGENT])
    result[SD.METADATA][SD.TIMESTEP] = \
        np.asarray([scenario_log_interval * i for i in range(episode_len)], dtype=np.float32)

    # Fill tracks
    all_objs = set()
    for frame in frames:
        all_objs.update(frame.step_info.keys())
    tracks = {
        k: dict(
            type=MetaDriveType.UNSET,
            state=dict(
                position=np.zeros(shape=(episode_len, 3)),
                size=np.zeros(shape=(episode_len, 3)),
                heading=np.zeros(shape=(episode_len, 1)),
                velocity=np.zeros(shape=(episode_len, 2)),
                valid=np.zeros(shape=(episode_len, 1))
            ),
            metadata=dict(track_length=episode_len, type=MetaDriveType.UNSET, object_id=k)
        )
        for k in list(all_objs)
    }
    for frame_idx in range(result[SD.LENGTH]):
        for id, state in frames[frame_idx].step_info.items():
            # Fill type
            tracks[id]["type"] = get_type_from_class(state["type"])
            tracks[id][SD.METADATA]["type"] = tracks[id]["type"]

            # Introducing the state item
            tracks[id]["state"]["position"][frame_idx] = state["position"]
            tracks[id]["state"]["heading"][frame_idx] = state["heading_theta"]
            tracks[id]["state"]["velocity"][frame_idx] = state["velocity"]
            tracks[id]["state"]["valid"][frame_idx] = 1
            if "size" in state:
                tracks[id]["state"]["size"][frame_idx] = state["size"]

        for id, policy_info in frames[frame_idx].policy_info.items():
            # Maybe actions is also recorded. If so, add item to tracks:
            if "action" in policy_info and policy_info["action"] is not {}:
                # TODO: In the case of discrete action, what should we do?
                action = np.asarray(policy_info["action"]).astype(np.float32)
                if "action" not in tracks[id]["state"]:
                    tracks[id]["state"]["action"] = np.zeros(shape=(episode_len, action.size), dtype=action.dtype)
                tracks[id]["state"]["action"][frame_idx] = action

    result[SD.TRACKS] = tracks

    # Traffic Light: Straight-through forward from original data
    result[SD.DYNAMIC_MAP_STATES] = {}  # old data has no traffic light info
    for k, manager_state in record_episode["manager_states"].items():
        if "DataManager" in k:
            if "raw_data" in manager_state:
                original_dynamic_map = copy.deepcopy(manager_state["raw_data"][SD.DYNAMIC_MAP_STATES])
                clipped_dynamic_map = {}
                for obj_id, obj_state in original_dynamic_map.items():
                    obj_state["state"] = {k: v[:episode_len] for k, v in obj_state["state"].items()}
                    clipped_dynamic_map[obj_id] = obj_state
                result[SD.METADATA]["history_metadata"] = manager_state["raw_data"][SD.METADATA]

    result = result.to_dict()
    SD.sanity_check(result, check_self_type=True)

    return result
