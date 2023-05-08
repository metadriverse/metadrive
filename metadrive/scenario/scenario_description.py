"""
A unified data format to describe a scenario that can be replayed by MetaDrive Simulator.

Example:

    scenario = {

        # ===== Meta data about the scenario =====
        # string. The name of the scenario
        "id": "Waymo-001",

        # string. The version of data format.
        "version": "MetaDrive v0.3.0.1",


        # int. The length of all trajectory and state arrays (T).
        "length": 200,

        # ===== Meta data ===
        "metadata": {

            # np.ndarray in (T, ). The time stamp of each time step.
            "ts": np.array([0.0, 0.1, 0.2, ...], dtype=np.float32),


            # bool. Whether the scenario is processed and exported by MetaDrive.
            # Some operations may be done, such as interpolating the lane to
            # make way points uniformly scattered in given interval.
            "metadrive_processed": True,

            # string. Coordinate system.
            "coordinate": "metadrive",

            # optional keys
            "source_file": "training_20s.tfrecord-00014-of-01000",
            "dataset": "waymo",
            "scenario_id": "dd0c8c27fdd6ef59",  # Used in Waymo dataset
            "seed": 512,
            "history_metadata": {},

            "sdc_id": "172",  # A key exists in tracks

        },

        # ===== Trajectories of active participants, e.g. vehicles, pedestrians =====
        # a dict mapping object ID to it's state dict.
        "tracks": {
            "vehicle1": {

                # The type string in metadrive.scenario.MetaDriveType
                "type": "VEHICLE",

                # The state dict. All values must have T elements.
                "state": {
                    "position": np.ones([200, 3], dtype=np.float32),
                    ...
                },

                # The meta data dict. Store useful information about the object. type in metadata could be those from
                # different dataset
                "metadata": {
                    "type": "VEHICLE",
                    "track_length": 200,
                    "object_id": "vehicle1",

                    # Optional keys
                    "agent_name": "default_agent",
                    "policy_spawn_info": {  # Information needed to re-instantiate the policy
                        "policy_class": ("metadrive.policy.idm_policy", "IDMPolicy"),
                        "args": ...,
                        "kwargs": ...,
                    }
                }
            },

            "pedestrian1": ...
        },

        # ===== States sequence of dynamics objects, e.g. traffic light =====
        # a dict mapping object ID to it's state dict.
        "dynamic_map_states": {
            "trafficlight1": {

                # The type string in metadrive.scenario.MetaDriveType
                "type": "TRAFFIC_LIGHT",

                # The state dict. All values must have T elements.
                "state": {
                    "object_state": np.ones([200, ], dtype=int),
                    ...
                },

                # The meta data dict. Store useful information about the object
                "metadata": {
                    "type": "TRAFFIC_LIGHT",
                    "track_length": 200,
                }
        }

        # ===== Map features =====
        # A dict mapping from map feature ID to a line segment
        "map_features": {
            "219": {
                "type": "LANE_SURFACE_STREET",
                "polyline": np.array in [21, 2],  # A set of 2D points describing a line segment
                # optional, only works for lane
                "polygon": np.array in [N, 2] # A set of 2D points representing convexhull
            },
            "182": ...
            ...
        }
    }
"""

import os
from collections import defaultdict

import numpy as np

from metadrive.type import MetaDriveType


class ScenarioDescription(dict):
    """
    MetaDrive Scenario Description. It stores keys of the data dict.
    """
    TRACKS = "tracks"
    VERSION = "version"
    ID = "id"
    DYNAMIC_MAP_STATES = "dynamic_map_states"
    MAP_FEATURES = "map_features"
    LENGTH = "length"
    METADATA = "metadata"
    FIRST_LEVEL_KEYS = {TRACKS, VERSION, ID, DYNAMIC_MAP_STATES, MAP_FEATURES, LENGTH, METADATA}

    # lane keys
    POLYLINE = "polyline"
    POLYGON = "polygon"
    LEFT_BOUNDARIES = "left_boundaries"
    RIGHT_BOUNDARIES = "right_boundaries"
    LEFT_NEIGHBORS = "left_neighbor"
    RIGHT_NEIGHBORS = "right_neighbor"
    ENTRY = "entry_lanes"
    EXIT = "exit_lanes"

    TYPE = "type"
    STATE = "state"
    OBJECT_ID = "object_id"
    STATE_DICT_KEYS = {TYPE, STATE, METADATA}
    ORIGINAL_ID_TO_OBJ_ID = "original_id_to_obj_id"
    OBJ_ID_TO_ORIGINAL_ID = "obj_id_to_original_id"
    TRAFFIC_LIGHT_POSITION = "stop_point"
    TRAFFIC_LIGHT_STATUS = "object_state"
    TRAFFIC_LIGHT_LANE = "lane"

    METADRIVE_PROCESSED = "metadrive_processed"
    TIMESTEP = "ts"
    COORDINATE = "coordinate"
    SDC_ID = "sdc_id"  # Not necessary, but can be stored in metadata.
    METADATA_KEYS = {METADRIVE_PROCESSED, COORDINATE, TIMESTEP}

    ALLOW_TYPES = (int, float, str, np.ndarray, dict, list, tuple, type(None), set)

    class SUMMARY:
        OBJECT_SUMMARY = "object_summary"
        NUMBER_SUMMARY = "number_summary"

        # for each object summary
        TYPE = "type"
        OBJECT_ID = "object_id"
        TRACK_LENGTH = "track_length"
        MOVING_DIST = "moving_distance"
        VALID_LENGTH = "valid_length"
        CONTINUOUS_VALID_LENGTH = "continuous_valid_length"

        # for number summary:
        NUM_OBJECTS = "num_objects"
        NUM_OBJECT_TYPES = "num_object_types"
        NUM_OBJECTS_EACH_TYPE = "num_objects_each_type"

        NUM_TRAFFIC_LIGHTS = "num_traffic_lights"
        NUM_TRAFFIC_LIGHT_TYPES = "num_traffic_light_types"
        NUM_TRAFFIC_LIGHTS_EACH_STEP = "num_traffic_light_each_step"

        NUM_MAP_FEATURES = "num_map_features"

    class DATASET:
        SUMMARY_FILE = "dataset_summary.pkl"  # dataset summary file name
        MAPPING_FILE = "dataset_mapping.pkl"  # store the relative path of summary file and each scenario

    @classmethod
    def sanity_check(cls, scenario_dict, check_self_type=False, valid_check=False):

        if check_self_type:
            assert isinstance(scenario_dict, dict)
            assert not isinstance(scenario_dict, ScenarioDescription)

        # Whether input has all required keys
        assert cls.FIRST_LEVEL_KEYS.issubset(set(scenario_dict.keys())), \
            "You lack these keys in first level: {}".format(cls.FIRST_LEVEL_KEYS.difference(set(scenario_dict.keys())))

        # Check types, only native python objects
        # This is to avoid issue in pickle deserialization
        _recursive_check_type(scenario_dict, cls.ALLOW_TYPES)

        scenario_length = scenario_dict[cls.LENGTH]

        # Check tracks data
        assert isinstance(scenario_dict[cls.TRACKS], dict)
        for obj_id, obj_state in scenario_dict[cls.TRACKS].items():
            cls._check_object_state_dict(
                obj_state, scenario_length=scenario_length, object_id=obj_id, valid_check=valid_check
            )

        # Check dynamic_map_state
        assert isinstance(scenario_dict[cls.DYNAMIC_MAP_STATES], dict)
        for obj_id, obj_state in scenario_dict[cls.DYNAMIC_MAP_STATES].items():
            cls._check_object_state_dict(obj_state, scenario_length=scenario_length, object_id=obj_id)

        # Check metadata
        assert isinstance(scenario_dict[cls.METADATA], dict)
        assert cls.METADATA_KEYS.issubset(set(scenario_dict[cls.METADATA].keys())), \
            "You lack these keys in metadata: {}".format(
                cls.METADATA_KEYS.difference(set(scenario_dict[cls.METADATA].keys()))
            )
        assert scenario_dict[cls.METADATA][cls.TIMESTEP].shape == (scenario_length, )

    @classmethod
    def _check_object_state_dict(cls, obj_state, scenario_length, object_id, valid_check=True):
        # Check keys
        assert set(obj_state).issuperset(cls.STATE_DICT_KEYS)

        # Check type
        assert MetaDriveType.has_type(obj_state[cls.TYPE]
                                      ), "MetaDrive doesn't have this type: {}".format(obj_state[cls.TYPE])

        # Check set type
        assert obj_state[cls.TYPE] != MetaDriveType.UNSET, "Types should be set for objects and traffic lights"

        # Check state arrays temporal consistency
        assert isinstance(obj_state[cls.STATE], dict)
        for state_key, state_array in obj_state[cls.STATE].items():
            assert isinstance(state_array, (np.ndarray, list, tuple))
            assert len(state_array) == scenario_length

            if not isinstance(state_array, np.ndarray):
                continue

            assert state_array.ndim in [1, 2], "Haven't implemented test array with dim {} yet".format(state_array.ndim)
            if state_array.ndim == 2:
                assert state_array.shape[
                    1] != 0, "Please convert all state with dim 1 to a 1D array instead of 2D array."

            if state_key == "valid" and valid_check:
                assert np.sum(state_array) >= 1, "No frame valid for this object. Consider removing it"

            # check valid
            if "valid" in obj_state[cls.STATE] and valid_check:
                _array = state_array[..., :2] if state_key == "position" else state_array
                assert abs(np.sum(_array[np.where(obj_state[cls.STATE]["valid"], False, True)])) < 1e-2, \
                    "Valid array mismatches with {} array, some frames in {} have non-zero values, " \
                    "so it might be valid".format(state_key, state_key)

        # Check metadata
        assert isinstance(obj_state[cls.METADATA], dict)
        for metadata_key in (cls.TYPE, cls.OBJECT_ID):
            assert metadata_key in obj_state[cls.METADATA]

        # Check metadata alignment
        if cls.OBJECT_ID in obj_state[cls.METADATA]:
            assert obj_state[cls.METADATA][cls.OBJECT_ID] == object_id

    def to_dict(self):
        return dict(self)

    def get_sdc_track(self):
        assert self.SDC_ID in self[self.METADATA]
        sdc_id = str(self[self.METADATA][self.SDC_ID])
        return self[self.TRACKS][sdc_id]

    @staticmethod
    def get_object_summary(state_dict, id, type):
        track = state_dict["position"]
        valid_track = track[np.where(state_dict["valid"].astype(int))][..., :2]
        distance = float(
            sum(np.linalg.norm(valid_track[i] - valid_track[i + 1]) for i in range(valid_track.shape[0] - 1))
        )
        valid_length = int(sum(state_dict["valid"]))

        continuous_valid_length = 0
        for v in state_dict["valid"]:
            if v:
                continuous_valid_length += 1
            if continuous_valid_length > 0 and not v:
                break

        return {
            ScenarioDescription.SUMMARY.TYPE: type,
            ScenarioDescription.SUMMARY.OBJECT_ID: str(id),
            ScenarioDescription.SUMMARY.TRACK_LENGTH: int(len(track)),
            ScenarioDescription.SUMMARY.MOVING_DIST: float(distance),
            ScenarioDescription.SUMMARY.VALID_LENGTH: int(valid_length),
            ScenarioDescription.SUMMARY.CONTINUOUS_VALID_LENGTH: int(continuous_valid_length)
        }

    @staticmethod
    def get_export_file_name(dataset, dataset_version, scenario_name):
        return "sd_{}_{}_{}.pkl".format(dataset, dataset_version, scenario_name)

    @staticmethod
    def is_scenario_file(file_name):
        file_name = os.path.basename(file_name)
        assert file_name[-4:] == ".pkl", "{} is not .pkl file".format(file_name)
        file_name = file_name.replace(".pkl", "")
        return os.path.basename(file_name)[:3] == "sd_" or all(char.isdigit() for char in file_name)

    @staticmethod
    def get_number_summary(scenario):
        number_summary_dict = {}
        # object
        number_summary_dict[ScenarioDescription.SUMMARY.NUM_OBJECTS] = len(scenario[ScenarioDescription.TRACKS])
        number_summary_dict[ScenarioDescription.SUMMARY.NUM_OBJECT_TYPES
                            ] = set(v["type"] for v in scenario[ScenarioDescription.TRACKS].values())
        object_types_counter = defaultdict(int)
        for v in scenario[ScenarioDescription.TRACKS].values():
            object_types_counter[v["type"]] += 1
        number_summary_dict[ScenarioDescription.SUMMARY.NUM_OBJECTS_EACH_TYPE] = dict(object_types_counter)

        # Number of different dynamic object states
        dynamic_object_states_types = set()
        dynamic_object_states_counter = defaultdict(int)
        for v in scenario[ScenarioDescription.DYNAMIC_MAP_STATES].values():
            for step_state in v["state"]["object_state"]:
                if step_state is None:
                    continue
                dynamic_object_states_types.add(step_state)
                dynamic_object_states_counter[step_state] += 1
        number_summary_dict[ScenarioDescription.SUMMARY.NUM_TRAFFIC_LIGHTS
                            ] = len(scenario[ScenarioDescription.DYNAMIC_MAP_STATES])
        number_summary_dict[ScenarioDescription.SUMMARY.NUM_TRAFFIC_LIGHT_TYPES] = dynamic_object_states_types
        number_summary_dict[ScenarioDescription.SUMMARY.NUM_TRAFFIC_LIGHTS_EACH_STEP
                            ] = dict(dynamic_object_states_counter)

        # map
        number_summary_dict[ScenarioDescription.SUMMARY.NUM_MAP_FEATURES
                            ] = len(scenario[ScenarioDescription.MAP_FEATURES])
        return number_summary_dict


def _recursive_check_type(obj, allow_types, depth=0):
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert isinstance(k, str), "Must use string to be dict keys"
            _recursive_check_type(v, allow_types, depth=depth + 1)

    if isinstance(obj, list):
        for v in obj:
            _recursive_check_type(v, allow_types, depth=depth + 1)

    assert isinstance(obj, allow_types), "Object type {} not allowed! ({})".format(type(obj), allow_types)

    if depth > 1000:
        raise ValueError()


# TODO (LQY): Remove me after paper writing
# {
#     "map_features": {
#         "map_object_id_1": {
#             "center_line": [...],
#             "polygon": [...],
#             "from_lanes": [...],
#             "to_lanes": [...],
#             "neighbor_lanes": [...],
#             "metadata": {...}
#         },
#         "map_object_id_2": {
#             "polyline": [...],
#             "type": "white_solid",
#             "metadata": {...}
#         }
#     },
#     "objects": {
#         "object_id_1": {
#             "position": [...],
#             "velocity": [...],
#             "heading": [...],
#             "size": [l, w, h],
#             "valid": [...],
#             "type": "VEHICLE",
#             "metadata": {...}
#         }
#     },
#     "traffic light": {
#         "light_id_1": {
#             "states": [...],
#             "position": [x, y, z],
#             "heading": -np.pi,
#             "lane_id": "lane_id_1",
#             "metadata": {...}
#         }
#     },
#     "metadata": {
#         "dataset": "nuscenes",
#         "episode_length": 198,
#         "time_interval": 0.1,
#         "sdc_id": "ego",
#         "coordinates": "right-hand"
#         ...
#     }
# }
#
# {
#     "map_features": {
#         "map_object_id_1": {
#             "center_line": [...],
#             "polygon": [...],
#             "connectivity": {...}
#         },
#         "map_object_id_2": {
#             "polyline": [...],
#             "type": "white_solid",
#         }
#     },
#     "objects": {
#         "object_id_1": {
#             "position": [...],
#             "heading": [...],
#             "type": "VEHICLE",
#         }
#     },
#     "traffic light": {
#         "light_id_1": {
#             "states": [...],
#             "lane_id": "lane_id_1",
#         }
#     },
#     "metadata": {
#         "dataset": "nuscenes",
#         "time_interval": 0.1,
#         "sdc_id": "ego",
#     }
# }
#
# {"map_features": {
# "map_object_id_1": {
#     "center_line": [...],
#     "type": "lane",
#     "connectivity": {...}},
# "map_object_id_2": {
#     "polyline": [...],
#     "type": "white_solid_line"}
# },
# "objects": {
#     "object_id_1": {
#         "position": [...],
#         "heading": [...],
#         "type": "VEHICLE"}
# },
# "traffic light": {
#     "light_id_1": {
#         "states": [...],
#         "lane": "map_object_id_1"}
# },
# "metadata": {
#     "dataset": "nuscenes",
#     "time_interval": 0.1
# }}
