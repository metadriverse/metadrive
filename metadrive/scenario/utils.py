import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure

from metadrive.component.traffic_participants.cyclist import Cyclist
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import DATA_VERSION, DEFAULT_AGENT
from metadrive.scenario import MetaDriveType, ScenarioDescription


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
    result = ScenarioDescription()

    result[ScenarioDescription.ID] = "{}-{}".format(
        record_episode["map_data"]["map_type"], record_episode["scenario_index"]
    )

    result[ScenarioDescription.METADRIVE_PROCESSED] = True

    result[ScenarioDescription.VERSION] = DATA_VERSION

    result["sdc_track_index"] = record_episode["frame"][0]._agent_to_object[DEFAULT_AGENT]

    result["map_features"] = record_episode["map_data"]["map_features"]

    scenario_log_interval = scenario_log_interval or record_episode["global_config"]["physics_world_step_size"]

    frames_skip = int(scenario_log_interval / record_episode["global_config"]["physics_world_step_size"])

    frames = [record_episode["frame"][i] for i in range(0, len(record_episode["frame"]), frames_skip)]

    length = len(frames)
    result[ScenarioDescription.LENGTH] = length

    result[ScenarioDescription.TIMESTEP] = np.asarray(
        [scenario_log_interval * i for i in range(length)], dtype=np.float32
    )

    # Fill tracks
    all_objs = set()
    for frame in frames:
        all_objs.update(frame.step_info.keys())
    tracks = {
        k: dict(
            type=MetaDriveType.UNSET,
            state=dict(
                position=np.zeros(shape=(length, 3)),
                size=np.zeros(shape=(length, 3)),
                heading=np.zeros(shape=(length, 1)),
                velocity=np.zeros(shape=(length, 2)),
                valid=np.zeros(shape=(length, 1))
            ),
            metadata=dict(track_length=length, type=MetaDriveType.UNSET, object_id=k)
        )
        for k in list(all_objs)
    }
    for frame_idx in range(len(result["ts"])):
        for id, state in frames[frame_idx].step_info.items():
            tracks[id]["type"] = get_type_from_class(state["type"])

            # Introducing the state item
            tracks[id]["state"]["position"][frame_idx] = state["position"]
            tracks[id]["state"]["heading"][frame_idx] = state["heading_theta"]
            tracks[id]["state"]["velocity"][frame_idx] = state["velocity"]
            tracks[id]["state"]["valid"][frame_idx] = 1
            if "size" in state:
                tracks[id]["state"]["size"][frame_idx] = state["size"]
    result[ScenarioDescription.TRACKS] = tracks

    # Traffic Light: Straight-through forward from original data
    result[ScenarioDescription.DYNAMIC_MAP_STATES] = {}  # old data has no traffic light info
    for k, manager_state in record_episode["manager_states"].items():
        if "DataManager" in k:
            if "raw_data" in manager_state:
                result[ScenarioDescription.DYNAMIC_MAP_STATES] = copy.deepcopy(
                    manager_state["raw_data"][ScenarioDescription.DYNAMIC_MAP_STATES]
                )

    ScenarioDescription.sanity_check(result)
    return result
