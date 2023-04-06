import copy

import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from metadrive.component.traffic_participants.cyclist import Cyclist
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import DATA_VERSION, DEFAULT_AGENT
from metadrive.scenario import ScenarioDescription as SD
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.type import MetaDriveType


def draw_map(map_features, show=False):
    figure(figsize=(8, 6), dpi=500)
    for key, value in map_features.items():
        if value.get("type", None) == MetaDriveType.LANE_SURFACE_STREET:
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
    elif issubclass(obj_class, BaseTrafficLight) or obj_class is BaseTrafficLight:
        return MetaDriveType.TRAFFIC_LIGHT
    else:
        return MetaDriveType.OTHER


def _convert_type_to_string(nested):
    if isinstance(nested, type):
        return (nested.__module__, nested.__name__)
    if isinstance(nested, (list, tuple)):
        return [_convert_type_to_string(v) for v in nested]
    if isinstance(nested, dict):
        return {k: _convert_type_to_string(v) for k, v in nested.items()}
    return nested


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

    result[SD.DYNAMIC_MAP_STATES] = {}

    # TODO: Fix this
    if scenario_log_interval != 0.1:
        raise ValueError("We don't support varying the scenario log interval yet.")

    frames = [step_frame_list[-1] for step_frame_list in record_episode["frame"]]

    episode_len = len(frames)
    assert frames[-1].episode_step == episode_len - 1, "Length mismatch"
    result[SD.LENGTH] = episode_len

    result[SD.METADATA] = {}
    result[SD.METADATA][SD.METADRIVE_PROCESSED] = True
    result[SD.METADATA]["dataset"] = "metadrive"
    result[SD.METADATA]["seed"] = record_episode["global_seed"]
    result[SD.METADATA]["scenario_id"] = record_episode["scenario_index"]
    result[SD.METADATA][SD.COORDINATE] = MetaDriveType.COORDINATE_METADRIVE
    result[SD.METADATA][SD.SDC_ID] = str(frames[0]._agent_to_object[DEFAULT_AGENT])
    result[SD.METADATA][SD.TIMESTEP] = \
        np.asarray([scenario_log_interval * i for i in range(episode_len)], dtype=np.float32)

    agent_to_object = {}
    object_to_agent = {}

    # Fill tracks
    all_objs = set()
    for frame in frames:
        all_objs.update(frame.step_info.keys())

    tracks = {
        k: dict(
            type=MetaDriveType.UNSET,
            state=dict(
                position=np.zeros(shape=(episode_len, 3)),
                heading=np.zeros(shape=(episode_len, 1)),
                velocity=np.zeros(shape=(episode_len, 2)),
                valid=np.zeros(shape=(episode_len, 1)),

                # Add these items when the object has them.
                # throttle_brake=np.zeros(shape=(episode_len, 1)),
                # steering=np.zeros(shape=(episode_len, 1)),
                # size=np.zeros(shape=(episode_len, 3)),
            ),
            metadata=dict(track_length=episode_len, type=MetaDriveType.UNSET, object_id=k, original_id=k)
        )
        for k in list(all_objs)
    }

    all_lights = set()
    # TODO LightManager may have different name
    for frame in frames:
        if "ScenarioLightManager" in frame.manager_info:
            all_lights.update(frame.manager_info["ScenarioLightManager"][SD.ORIGINAL_ID_TO_OBJ_ID].keys())

    lights = {
        k: dict(
            type=MetaDriveType.TRAFFIC_LIGHT,
            state={
                ScenarioDescription.TRAFFIC_LIGHT_POSITION: np.zeros(shape=(episode_len, 2)),
                ScenarioDescription.TRAFFIC_LIGHT_STATUS: np.array([MetaDriveType.LIGHT_UNKNOWN for _ in range(episode_len)]),
                ScenarioDescription.TRAFFIC_LIGHT_LANE: np.zeros(shape=(episode_len,)),
            },
            metadata=dict(track_length=episode_len,
                          type=MetaDriveType.TRAFFIC_LIGHT,
                          object_id=k,
                          dataset="metadrive")
        )
        for k in list(all_lights)
    }

    for frame_idx in range(result[SD.LENGTH]):

        # Record all agents' states (position, velocity, ...)
        for id, state in frames[frame_idx].step_info.items():
            # Fill type
            type = get_type_from_class(state["type"])
            if type == MetaDriveType.TRAFFIC_LIGHT:
                # pop id from tracks
                if id in tracks:
                    tracks.pop(id)
                assert "ScenarioLightManager" in frames[frame_idx].manager_info, "Can not find light manager info"

                # convert to original id
                id = frames[frame_idx].manager_info["ScenarioLightManager"][SD.OBJ_ID_TO_ORIGINAL_ID][id]

                lights[id]["type"] = type
                lights[id][SD.METADATA]["type"] = lights[id]["type"]

                # Introducing the state item
                light_status = state[ScenarioDescription.TRAFFIC_LIGHT_STATUS]
                lights[id]["state"][ScenarioDescription.TRAFFIC_LIGHT_STATUS][frame_idx] = light_status
                if light_status != MetaDriveType.LIGHT_UNKNOWN:
                    lights[id]["state"][ScenarioDescription.TRAFFIC_LIGHT_LANE][frame_idx] = int(id)
                    lights[id]["state"][ScenarioDescription.TRAFFIC_LIGHT_POSITION][frame_idx] = state[
                        ScenarioDescription.TRAFFIC_LIGHT_POSITION]
            else:
                tracks[id]["type"] = type
                tracks[id][SD.METADATA]["type"] = tracks[id]["type"]

                # Introducing the state item
                tracks[id]["state"]["position"][frame_idx] = state["position"]
                tracks[id]["state"]["heading"][frame_idx] = state["heading_theta"]
                tracks[id]["state"]["velocity"][frame_idx] = state["velocity"]
                tracks[id]["state"]["valid"][frame_idx] = 1

                if "throttle_brake" in state:
                    if "throttle_brake" not in tracks[id]["state"]:
                        tracks[id]["state"]["throttle_brake"] = np.zeros(shape=(episode_len, 1))
                    tracks[id]["state"]["throttle_brake"][frame_idx] = state["throttle_brake"]

                if "steering" in state:
                    if "steering" not in tracks[id]["state"]:
                        tracks[id]["state"]["steering"] = np.zeros(shape=(episode_len, 1))
                    tracks[id]["state"]["steering"][frame_idx] = state["steering"]

                if "length" in state:
                    if "length" not in tracks[id]["state"]:
                        tracks[id]["state"]["length"] = np.zeros(shape=(episode_len, 1))
                    tracks[id]["state"]["length"][frame_idx] = state["length"]

                if "width" in state:
                    if "width" not in tracks[id]["state"]:
                        tracks[id]["state"]["width"] = np.zeros(shape=(episode_len, 1))
                    tracks[id]["state"]["width"][frame_idx] = state["width"]

                if "height" in state:
                    if "height" not in tracks[id]["state"]:
                        tracks[id]["state"]["height"] = np.zeros(shape=(episode_len, 1))
                    tracks[id]["state"]["height"][frame_idx] = state["height"]

                if id in frames[frame_idx]._object_to_agent:
                    tracks[id]["metadata"]["agent_name"] = frames[frame_idx]._object_to_agent[id]

                # TODO Manager may have different name
                if "ScenarioTrafficManager" in frames[frame_idx].manager_info:
                    origin_id = frames[frame_idx].manager_info["ScenarioTrafficManager"][SD.OBJ_ID_TO_ORIGINAL_ID][id]
                    if tracks[id]["metadata"]["original_id"] == id:
                        tracks[id]["metadata"]["original_id"] = origin_id
                    else:
                        assert tracks[id]["metadata"]["original_id"] == origin_id

        # Record all policies' states (action, ...)
        for id, policy_info in frames[frame_idx].policy_info.items():
            # Maybe actions is also recorded. If so, add item to tracks:
            # TODO: In the case of discrete action, what should we do?
            for key, policy_state in policy_info.items():
                if policy_state is {}:
                    continue
                policy_state = np.asarray(policy_state)
                assert policy_state.dtype != object
                if key not in tracks[id]["state"]:
                    tracks[id]["state"][key] = np.zeros(
                        shape=(episode_len, policy_state.size), dtype=policy_state.dtype
                    )
                tracks[id]["state"][key][frame_idx] = policy_state

        # Record policy metadata
        for id, policy_spawn_info in frames[frame_idx].policy_spawn_info.items():
            tracks[id]["metadata"]["policy_spawn_info"] = copy.deepcopy(_convert_type_to_string(policy_spawn_info))

        # Record agent2object, object2agent mapping for metadata
        agent_to_object.update(frames[frame_idx]._agent_to_object)
        object_to_agent.update(frames[frame_idx]._object_to_agent)

        # Record spawn information
        for obj_name, spawn_info in frames[frame_idx].spawn_info.items():
            if obj_name not in tracks:
                continue
            spawn_info = copy.deepcopy(spawn_info)
            spawn_info = _convert_type_to_string(spawn_info)
            if "config" in spawn_info:
                spawn_info.pop("config")
            tracks[obj_name]["metadata"]["spawn_info"] = spawn_info

    result[SD.TRACKS] = tracks
    result[SD.DYNAMIC_MAP_STATES] = lights

    # # Traffic Light: Straight-through forward from original data
    # for k, manager_state in record_episode["manager_metadata"].items():
    #     if "DataManager" in k:
    #         if "raw_data" in manager_state:
    #             original_dynamic_map = copy.deepcopy(manager_state["raw_data"][SD.DYNAMIC_MAP_STATES])
    #             clipped_dynamic_map = {}
    #             for obj_id, obj_state in original_dynamic_map.items():
    #                 obj_state["state"] = {k: v[:episode_len] for k, v in obj_state["state"].items()}
    #                 clipped_dynamic_map[obj_id] = obj_state
    #             result[SD.METADATA]["history_metadata"] = manager_state["raw_data"][SD.METADATA]
    #             result[SD.DYNAMIC_MAP_STATES] = clipped_dynamic_map

    # Record agent2object, object2agent metadata
    result[SD.METADATA]["agent_to_object"] = {str(k): str(v) for k, v in agent_to_object.items()}
    result[SD.METADATA]["object_to_agent"] = {str(k): str(v) for k, v in object_to_agent.items()}

    result = result.to_dict()
    SD.sanity_check(result, check_self_type=True)

    return result


def read_scenario_data(file_path):
    with open(file_path, "rb") as f:
        # unpickler = CustomUnpickler(f)
        data = pickle.load(f)
    data = ScenarioDescription(data)
    return data


def convert_polyline_to_metadrive(polyline, coordinate_transform=True):
    """
    Convert a polyline to metadrive coordinate as np.array
    """
    polyline = np.asarray(polyline)
    if coordinate_transform:
        return np.stack([polyline[:, 0], -polyline[:, 1]], axis=1)
    else:
        return polyline
