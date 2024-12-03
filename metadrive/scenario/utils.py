import copy
import os
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure

from metadrive.component.static_object.traffic_object import TrafficCone, TrafficBarrier
from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
from metadrive.component.traffic_participants.cyclist import Cyclist
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import DATA_VERSION, DEFAULT_AGENT
from metadrive.engine import get_logger
from metadrive.scenario import ScenarioDescription as SD
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.type import MetaDriveType
from metadrive.utils.math import wrap_to_pi

NP_ARRAY_DECIMAL = 3
VELOCITY_DECIMAL = 1  # velocity can not be set accurately
MIN_LENGTH_RATIO = 0.8

logger = get_logger()


def dict_recursive_remove_array_and_set(d):
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, set):
        return tuple(d)
    if isinstance(d, dict):
        for k in d.keys():
            d[k] = dict_recursive_remove_array_and_set(d[k])
    return d


def draw_map(map_features, show=False):
    figure(figsize=(8, 6), dpi=500)
    for key, value in map_features.items():
        if MetaDriveType.is_lane(value.get("type", None)):
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
    elif issubclass(obj_class, TrafficBarrier) or obj_class is TrafficBarrier:
        return MetaDriveType.TRAFFIC_BARRIER
    elif issubclass(obj_class, TrafficCone) or obj_class is TrafficCone:
        return MetaDriveType.TRAFFIC_CONE
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


def find_light_manager_name(manager_info):
    """
    Find the light_manager in real data manager
    """
    for manager_name in manager_info:
        if "LightManager" in manager_name:
            return manager_name
    return None


def find_traffic_manager_name(manager_info):
    """
    Find the traffic_manager in real data manager
    """
    for manager_name in manager_info:
        if "TrafficManager" in manager_name and manager_name != "PGTrafficManager":
            return manager_name
    return None


def find_data_manager_name(manager_info):
    """
    Find the data_manager
    """
    for manager_name in manager_info:
        if "DataManager" in manager_name:
            return manager_name
    return None


def convert_recorded_scenario_exported(record_episode, scenario_log_interval=0.1, to_dict=True):
    """
    This function converts the internal run data of MetaDrive to the ScenarioNet's scenario description.

    The output data structure follows MetaDrive data format, but with some changes compared to the original data.
    For example, MetaDrive InterpolateLane will reformat the Lane data and making all waypoints equally distancing.
    We call this lane sampling rate, which is 0.2m in MetaDrive but might different in other datasets.

    Args:
        record_episode: the internal data structure from MetaDrive run.
        scenario_log_interval: the time interval for one step.
        to_dict: whether to return a python dict or a ScenarioDescription object.

    Returns:
        a python dict or a ScenarioDescription object.
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
    result[SD.METADATA][SD.ID] = result[SD.ID]
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

    traffic_manager_name = find_traffic_manager_name(record_episode["manager_metadata"])
    light_manager_name = find_light_manager_name(record_episode["manager_metadata"])
    data_manager_name = find_data_manager_name(record_episode["manager_metadata"])

    tracks = {}
    for k in list(all_objs):
        tracks[k] = dict(
            type=MetaDriveType.UNSET,
            state=dict(
                position=np.zeros(shape=(episode_len, 3)),
                heading=np.zeros(shape=(episode_len, )),
                velocity=np.zeros(shape=(episode_len, 2)),
                valid=np.zeros(shape=(episode_len, )),

                # Add these items when the object has them.
                # throttle_brake=np.zeros(shape=(episode_len, 1)),
                # steering=np.zeros(shape=(episode_len, 1)),
                # size=np.zeros(shape=(episode_len, 3)),
            ),
            metadata=dict(track_length=episode_len, type=MetaDriveType.UNSET, object_id=k, original_id=k)
        )

    all_lights = set()
    if light_manager_name is not None:
        for frame in frames:
            all_lights.update(frame.manager_info[light_manager_name][SD.ORIGINAL_ID_TO_OBJ_ID].keys())

    lights = {
        k: {
            "type": MetaDriveType.TRAFFIC_LIGHT,
            "state": {
                ScenarioDescription.TRAFFIC_LIGHT_STATUS: [MetaDriveType.LIGHT_UNKNOWN] * episode_len
            },
            ScenarioDescription.TRAFFIC_LIGHT_POSITION: np.zeros(shape=(3, ), dtype=np.float32),
            ScenarioDescription.TRAFFIC_LIGHT_LANE: None,
            "metadata": dict(
                track_length=episode_len, type=MetaDriveType.TRAFFIC_LIGHT, object_id=k, dataset="metadrive"
            )
        }
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
                assert light_manager_name in frames[frame_idx].manager_info, "Can not find light manager info"

                # convert to original id
                id = frames[frame_idx].manager_info[light_manager_name][SD.OBJ_ID_TO_ORIGINAL_ID][id]

                lights[id]["type"] = type
                lights[id][SD.METADATA]["type"] = lights[id]["type"]

                # Introducing the state item
                light_status = state[ScenarioDescription.TRAFFIC_LIGHT_STATUS]
                lights[id]["state"][ScenarioDescription.TRAFFIC_LIGHT_STATUS][frame_idx] = light_status

                # if light_status != MetaDriveType.LIGHT_UNKNOWN:
                if lights[id][ScenarioDescription.TRAFFIC_LIGHT_LANE] is None:
                    lights[id][ScenarioDescription.TRAFFIC_LIGHT_LANE] = str(id)
                    lights[id][ScenarioDescription.TRAFFIC_LIGHT_POSITION
                               ] = state[ScenarioDescription.TRAFFIC_LIGHT_POSITION]
                else:
                    assert lights[id][ScenarioDescription.TRAFFIC_LIGHT_LANE] == str(id)
                    assert lights[id][ScenarioDescription.TRAFFIC_LIGHT_POSITION
                                      ] == state[ScenarioDescription.TRAFFIC_LIGHT_POSITION]

            else:
                tracks[id]["type"] = type
                tracks[id][SD.METADATA]["type"] = tracks[id]["type"]

                # Introducing the state item
                tracks[id]["state"]["position"][frame_idx] = state["position"]
                tracks[id]["state"]["heading"][frame_idx] = state["heading_theta"]
                tracks[id]["state"]["velocity"][frame_idx] = state["velocity"]
                if "width" in state:
                    if "width" not in tracks[id]["state"]:
                        tracks[id]["state"]["width"] = np.zeros(shape=(episode_len, 1))
                    tracks[id]["state"]["width"][frame_idx] = state["width"]
                if "length" in state:
                    if "length" not in tracks[id]["state"]:
                        tracks[id]["state"]["length"] = np.zeros(shape=(episode_len, 1))
                    tracks[id]["state"]["length"][frame_idx] = state["length"]
                if "height" in state:
                    if "height" not in tracks[id]["state"]:
                        tracks[id]["state"]["height"] = np.zeros(shape=(episode_len, 1))
                    tracks[id]["state"]["height"][frame_idx] = state["height"]
                if "radius" in state:
                    if "radius" not in tracks[id]["state"]:
                        tracks[id]["state"]["radius"] = np.zeros(shape=(episode_len, 1))
                    tracks[id]["state"]["radius"][frame_idx] = state["radius"]
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

                if traffic_manager_name is not None:
                    origin_id = frames[frame_idx].manager_info[traffic_manager_name][SD.OBJ_ID_TO_ORIGINAL_ID][id]
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

    # Record agent2object, object2agent metadata
    result[SD.METADATA]["agent_to_object"] = {str(k): str(v) for k, v in agent_to_object.items()}
    result[SD.METADATA]["object_to_agent"] = {str(k): str(v) for k, v in object_to_agent.items()}

    if data_manager_name is not None:
        data_manager_raw_data = record_episode["manager_metadata"][data_manager_name].get("raw_data", None)
        if data_manager_raw_data:
            result[SD.METADATA]["history_metadata"] = data_manager_raw_data["metadata"]
    if to_dict:
        result = result.to_dict()
        SD.sanity_check(result, check_self_type=True)

    return result


def read_scenario_data(file_path, centralize=False):
    """Read a scenario pkl file and return the Scenario Description instance.

    Args:
        file_path: the path to a scenario file (usually ends with `.pkl`).
        centralize: whether to centralize all elements to the ego car's initial position

    Returns:
        The Scenario Description instance of that scenario.
    """
    assert SD.is_scenario_file(file_path), "File: {} is not scenario file".format(file_path)
    with open(file_path, "rb") as f:
        # unpickler = CustomUnpickler(f)
        data = pickle.load(f)
    data = ScenarioDescription(data)
    if centralize:
        data = ScenarioDescription.centralize_to_ego_car_initial_position(data)
    return data


def read_dataset_summary(file_folder, check_file_existence=True):
    """
    Read the `dataset_summary.pkl` and return the metadata of each scenario in this dataset.

    This function supports two methods to load pickle files. The first is the old method where we store pickle files in
    0.pkl, 1.pkl, .... The second is the new method which use a summary file to record important metadata of
    each scenario.

    Args:
        file_folder: the path to the root folder of your dataset.
        check_file_existence: check if all scenarios registered in the summary file exist.

    Returns:
        A tuple of three elements:
        1) the summary dict mapping from scenario ID to its metadata,
        2) the list of all scenarios IDs, and
        3) a dict mapping from scenario IDs to the folder that hosts their files.
    """
    file_folder = pathlib.Path(file_folder)
    summary_file = file_folder / SD.DATASET.SUMMARY_FILE
    mapping_file = file_folder / SD.DATASET.MAPPING_FILE
    if os.path.isfile(summary_file):
        with open(summary_file, "rb") as f:
            summary_dict = pickle.load(f)

    else:
        logger.warning(f"Summary file is not found at {summary_file}! Generate a dummy one.")

        # === The following is deprecated ===
        # Create a fake one
        files = []
        for file in os.listdir(file_folder):
            if SD.is_scenario_file(os.path.basename(file)):
                files.append(file)
        summary_dict = {f: {} for f in files}

    mapping = None
    if os.path.exists(mapping_file):
        with open(mapping_file, "rb") as f:
            mapping = pickle.load(f)

    if not mapping:
        # Create a fake one
        mapping = {k: "" for k in summary_dict}

    if check_file_existence:
        for file in summary_dict:
            assert file in mapping, "FileName in mapping mismatch with summary"
            assert SD.is_scenario_file(file), "File:{} is not sd scenario file".format(file)
            file_path = file_folder / mapping[file] / file
            assert os.path.exists(file_path), "Can not find file: {}".format(file_path)

    return summary_dict, list(summary_dict.keys()), mapping


def extract_dataset_summary_and_mapping(scenario_list, dataset_name, dataset_version):
    """Extract the dataset summary and mapping for dataset_summary.pkl and dataset_mapping.pkl.

    Args:
        scenario_list: A list of Scenario Description objects.

    Returns:
        Summary dict, mapping dict, scenario dict
    """
    summary = {}
    mapping = {}
    scenario_dict = {}
    for sd_scenario in scenario_list:
        scenario_id = sd_scenario[SD.ID]
        export_file_name = SD.get_export_file_name(dataset_name, dataset_version, scenario_id)

        sd_scenario = SD(sd_scenario)
        if hasattr(SD, "update_summaries"):
            SD.update_summaries(sd_scenario)
        else:
            raise ValueError("Please update MetaDrive to latest version.")

        # update summary/mapping dict
        if export_file_name in summary:
            logger.warning("Scenario {} already exists and will be overwritten!".format(export_file_name))
        summary[export_file_name] = copy.deepcopy(sd_scenario[SD.METADATA])
        mapping[export_file_name] = ""  # in the same dir

        # sanity check
        sd_scenario = sd_scenario.to_dict()
        SD.sanity_check(sd_scenario, check_self_type=True)

        scenario_dict[export_file_name] = sd_scenario

    return summary, mapping, scenario_dict


def save_dataset(scenario_list, dataset_name, dataset_version, dataset_dir):
    summary, mapping, scenario_dict = extract_dataset_summary_and_mapping(scenario_list, dataset_name, dataset_version)
    summary_file = pathlib.Path(dataset_dir) / ScenarioDescription.DATASET.SUMMARY_FILE
    summary_file = summary_file.resolve()
    mapping_file = pathlib.Path(dataset_dir) / ScenarioDescription.DATASET.MAPPING_FILE
    mapping_file = mapping_file.resolve()
    os.makedirs(dataset_dir, exist_ok=True)
    for file_name, scenario in scenario_dict.items():
        file_path = pathlib.Path(dataset_dir) / file_name
        with open(file_path, "wb") as file:
            pickle.dump(scenario, file)
    with open(summary_file, "wb") as file:
        pickle.dump(dict_recursive_remove_array_and_set(summary), file)
    with open(mapping_file, "wb") as file:
        pickle.dump(mapping, file)
    print(
        "\n ================ Dataset Summary and Mapping are saved at: {} "
        "================ \n".format(summary_file)
    )


def get_number_of_scenarios(dataset_path):
    _, files, _ = read_dataset_summary(dataset_path)
    return len(files)


def assert_scenario_equal(scenarios1, scenarios2, only_compare_sdc=False, check_self_type=True):
    # ===== These two set of data should align =====
    assert set(scenarios1.keys()) == set(scenarios2.keys())
    for scenario_id in scenarios1.keys():
        SD.sanity_check(scenarios1[scenario_id], check_self_type=check_self_type)
        SD.sanity_check(scenarios2[scenario_id], check_self_type=check_self_type)
        old_scene = SD(scenarios1[scenario_id])
        new_scene = SD(scenarios2[scenario_id])
        SD.sanity_check(old_scene)
        SD.sanity_check(new_scene)
        assert old_scene[SD.LENGTH] >= new_scene[SD.LENGTH], (old_scene[SD.LENGTH], new_scene[SD.LENGTH])

        if only_compare_sdc:
            sdc1 = old_scene[SD.METADATA][SD.SDC_ID]
            sdc2 = new_scene[SD.METADATA][SD.SDC_ID]
            state_dict1 = old_scene[SD.TRACKS][sdc1]
            state_dict2 = new_scene[SD.TRACKS][sdc2]
            min_len = min(state_dict1[SD.STATE]["position"].shape[0], state_dict2[SD.STATE]["position"].shape[0])
            max_len = max(state_dict1[SD.STATE]["position"].shape[0], state_dict2[SD.STATE]["position"].shape[0])
            assert min_len / max_len > MIN_LENGTH_RATIO, "Replayed Scenario length ratio: {}".format(min_len / max_len)
            for k in state_dict1[SD.STATE].keys():
                if k in ["action", "throttle_brake", "steering"]:
                    continue
                elif k == "position":
                    np.testing.assert_almost_equal(
                        state_dict1[SD.STATE][k][:min_len][..., :2],
                        state_dict2[SD.STATE][k][:min_len][..., :2],
                        decimal=NP_ARRAY_DECIMAL
                    )
                elif k == "heading":
                    np.testing.assert_almost_equal(
                        wrap_to_pi(state_dict1[SD.STATE][k][:min_len] - state_dict2[SD.STATE][k][:min_len]),
                        np.zeros_like(state_dict2[SD.STATE][k][:min_len]),
                        decimal=NP_ARRAY_DECIMAL
                    )
                elif k == "velocity":
                    np.testing.assert_almost_equal(
                        state_dict1[SD.STATE][k][:min_len],
                        state_dict2[SD.STATE][k][:min_len],
                        decimal=VELOCITY_DECIMAL
                    )
            assert state_dict1[SD.TYPE] == state_dict2[SD.TYPE]

        else:
            # assert set(old_scene[SD.TRACKS].keys()).issuperset(set(new_scene[SD.TRACKS].keys()) - {new_scene[SD.METADATA][SD.SDC_ID]})
            assert len(old_scene[SD.TRACKS]) == len(new_scene[SD.TRACKS]), "obj num mismatch"
            for track_id, track in old_scene[SD.TRACKS].items():
                if track_id == new_scene[SD.METADATA][SD.SDC_ID]:
                    continue
                if track_id not in new_scene[SD.TRACKS]:
                    assert track_id == old_scene[SD.METADATA][SD.SDC_ID]
                    continue
                for state_k in new_scene[SD.TRACKS][track_id][SD.STATE]:
                    state_array_1 = new_scene[SD.TRACKS][track_id][SD.STATE][state_k]
                    state_array_2 = track[SD.STATE][state_k]
                    min_len = min(state_array_1.shape[0], state_array_2.shape[0])
                    max_len = max(state_array_1.shape[0], state_array_2.shape[0])
                    assert min_len / max_len > MIN_LENGTH_RATIO, "Replayed Scenario length ratio: {}".format(
                        min_len / max_len
                    )

                    if state_k == "velocity":
                        decimal = VELOCITY_DECIMAL
                    else:
                        decimal = NP_ARRAY_DECIMAL

                    if state_k == "heading":
                        # error < 5.7 degree is acceptable
                        broader_ratio = 1
                        ret = abs(wrap_to_pi(state_array_1[:min_len] - state_array_2[:min_len])) < 1e-1
                        ratio = np.sum(np.asarray(ret, dtype=np.int8)) / len(ret)
                        if ratio < broader_ratio:
                            raise ValueError("Match ration: {}, Target: {}".format(ratio, broader_ratio))

                        strict_ratio = 0.98
                        ret = abs(wrap_to_pi(state_array_1[:min_len] - state_array_2[:min_len])) < 1e-4
                        ratio = np.sum(np.asarray(ret, dtype=np.int8)) / len(ret)
                        if ratio < strict_ratio:
                            raise ValueError("Match ration: {}, Target: {}".format(ratio, strict_ratio))
                    else:
                        strict_ratio = 0.99
                        ret = abs(wrap_to_pi(state_array_1[:min_len] - state_array_2[:min_len])) < pow(10, -decimal)
                        ratio = np.sum(np.asarray(ret, dtype=np.int8)) / len(ret)
                        if ratio < strict_ratio:
                            raise ValueError("Match ration: {}, Target: {}".format(ratio, strict_ratio))

                assert new_scene[SD.TRACKS][track_id][SD.TYPE] == track[SD.TYPE]

            track_id = new_scene[SD.METADATA][SD.SDC_ID]
            for k in new_scene.get_sdc_track()["state"]:
                state_array_1 = new_scene.get_sdc_track()["state"][k]
                state_array_2 = old_scene.get_sdc_track()["state"][k]
                min_len = min(state_array_1.shape[0], state_array_2.shape[0])
                max_len = max(state_array_1.shape[0], state_array_2.shape[0])
                assert min_len / max_len > MIN_LENGTH_RATIO, "Replayed Scenario length ratio: {}".format(
                    min_len / max_len
                )

                if k == "velocity":
                    decimal = VELOCITY_DECIMAL
                elif k == "position":
                    state_array_1 = state_array_1[..., :2]
                    state_array_2 = state_array_2[..., :2]
                    decimal = NP_ARRAY_DECIMAL
                else:
                    decimal = NP_ARRAY_DECIMAL
                np.testing.assert_almost_equal(state_array_1[:min_len], state_array_2[:min_len], decimal=decimal)

            assert new_scene[SD.TRACKS][track_id][SD.TYPE] == track[SD.TYPE]

        assert set(old_scene[SD.MAP_FEATURES].keys()).issuperset(set(new_scene[SD.MAP_FEATURES].keys()))
        assert set(old_scene[SD.DYNAMIC_MAP_STATES].keys()) == set(new_scene[SD.DYNAMIC_MAP_STATES].keys())

        for map_id, map_feat in old_scene[SD.MAP_FEATURES].items():
            # It is possible that some line are not included in new scene but exist in old scene.
            # old_scene_polyline = map_feat["polyline"]
            # if coordinate_transform:
            #     old_scene_polyline = waymo_to_metadrive_vector(old_scene_polyline)
            if map_feat["type"] == MetaDriveType.LANE_SURFACE_STREET:
                assert len(map_feat["polyline"]) - len(new_scene[SD.MAP_FEATURES][map_id]["polyline"]) <= 1
                _min = min(len(map_feat["polyline"]), len(new_scene[SD.MAP_FEATURES][map_id]["polyline"]))
                np.testing.assert_almost_equal(
                    new_scene[SD.MAP_FEATURES][map_id]["polyline"][:_min], map_feat["polyline"][:_min], decimal=1
                )
            assert new_scene[SD.MAP_FEATURES][map_id][SD.TYPE] == map_feat[SD.TYPE]

        for obj_id, obj_state in old_scene[SD.DYNAMIC_MAP_STATES].items():
            new_state_dict = new_scene[SD.DYNAMIC_MAP_STATES][obj_id][SD.STATE]
            old_state_dict = obj_state[SD.STATE]
            assert set(new_state_dict.keys()) == set(old_state_dict.keys())
            for k in new_state_dict.keys():
                min_len = min(new_state_dict[k].shape[0], old_state_dict[k].shape[0])
                max_len = max(new_state_dict[k].shape[0], old_state_dict[k].shape[0])
                assert min_len / max_len > MIN_LENGTH_RATIO, "Replayed Scenario length ratio: {}".format(
                    min_len / max_len
                )
                if k == ScenarioDescription.TRAFFIC_LIGHT_STATUS:
                    same_light = new_state_dict[k][:min_len] == old_state_dict[k][:min_len]
                    assert same_light.all()
                else:
                    np.testing.assert_almost_equal(
                        new_state_dict[k][:min_len], old_state_dict[k][:min_len], decimal=NP_ARRAY_DECIMAL
                    )

            assert new_scene[SD.DYNAMIC_MAP_STATES][obj_id][SD.TYPE] == obj_state[SD.TYPE]
