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

                # The meta data dict. Store useful information about the object
                "metadata": {
                    "type": "VEHICLE",
                    "track_length": 200
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
    }
"""
import numpy as np

from metadrive.scenario.metadrive_type import MetaDriveType


class ScenarioDescription(dict):
    """
    MetaDrive Scenario Description
    """
    TRACKS = "tracks"
    VERSION = "version"
    ID = "id"
    DYNAMIC_MAP_STATES = "dynamic_map_states"
    MAP_FEATURES = "map_features"
    LENGTH = "length"
    METADATA = "metadata"
    FIRST_LEVEL_KEYS = {TRACKS, VERSION, ID, DYNAMIC_MAP_STATES, MAP_FEATURES, LENGTH, METADATA}

    TYPE = "type"
    STATE = "state"
    OBJECT_ID = "object_id"
    STATE_DICT_KEYS = {TYPE, STATE, METADATA}

    METADRIVE_PROCESSED = "metadrive_processed"
    TIMESTEP = "ts"
    COORDINATE = "coordinate"
    SDC_ID = "sdc_id"  # Not necessary, but can be stored in metadata.
    CREATED_TIME = "created_time"
    METADATA_KEYS = {METADRIVE_PROCESSED, COORDINATE, TIMESTEP, CREATED_TIME}

    ALLOW_TYPES = (int, float, str, np.ndarray, dict, list)

    @classmethod
    def sanity_check(cls, scenario_dict):
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
            cls._check_object_state_dict(obj_state, scenario_length=scenario_length, object_id=obj_id)

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
    def _check_object_state_dict(cls, obj_state, scenario_length, object_id):
        # Check keys
        assert set(obj_state) == cls.STATE_DICT_KEYS

        # Check type
        assert MetaDriveType.has_type(obj_state[cls.TYPE]
                                      ), "MetaDrive doesn't have this type: {}".format(obj_state[cls.TYPE])

        # Check state arrays temporal consistency
        assert isinstance(obj_state[cls.STATE], dict)
        for state_key, state_array in obj_state[cls.STATE].items():
            assert isinstance(state_array, np.ndarray)
            assert state_array.shape[0] == scenario_length

        # Check metadata
        assert isinstance(obj_state[cls.METADATA], dict)
        for metadata_key in (cls.TYPE, cls.OBJECT_ID):
            assert metadata_key in obj_state[cls.METADATA]

        # Check metadata alignment
        if cls.OBJECT_ID in obj_state[cls.METADATA]:
            assert obj_state[cls.METADATA][cls.OBJECT_ID] == object_id


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
