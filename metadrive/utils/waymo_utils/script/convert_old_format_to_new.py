"""
I only use this script to convert 3 old pickle cases into new format. People should generate the dara by using
convert_waymo_to_metadrive.py
"""
import pickle
from enum import Enum

from metadrive.engine.asset_loader import AssetLoader


class LaneTypeClass:
    UNKNOWN = 0
    LANE_FREEWAY = 1
    LANE_SURFACE_STREET = 2
    LANE_BIKE_LANE = 3


class RoadLineType(Enum):
    UNKNOWN = 0
    BROKEN_SINGLE_WHITE = 1
    SOLID_SINGLE_WHITE = 2
    SOLID_DOUBLE_WHITE = 3
    BROKEN_SINGLE_YELLOW = 4
    BROKEN_DOUBLE_YELLOW = 5
    SOLID_SINGLE_YELLOW = 6
    SOLID_DOUBLE_YELLOW = 7
    PASSING_DOUBLE_YELLOW = 8

    ENUM_TO_STR = {
        UNKNOWN: 'UNKNOWN',
        BROKEN_SINGLE_WHITE: 'ROAD_LINE_BROKEN_SINGLE_WHITE',
        SOLID_SINGLE_WHITE: 'ROAD_LINE_SOLID_SINGLE_WHITE',
        SOLID_DOUBLE_WHITE: 'ROAD_LINE_SOLID_DOUBLE_WHITE',
        BROKEN_SINGLE_YELLOW: 'ROAD_LINE_BROKEN_SINGLE_YELLOW',
        BROKEN_DOUBLE_YELLOW: 'ROAD_LINE_BROKEN_DOUBLE_YELLOW',
        SOLID_SINGLE_YELLOW: 'ROAD_LINE_SOLID_SINGLE_YELLOW',
        SOLID_DOUBLE_YELLOW: 'ROAD_LINE_SOLID_DOUBLE_YELLOW',
        PASSING_DOUBLE_YELLOW: 'ROAD_LINE_PASSING_DOUBLE_YELLOW'
    }

    @staticmethod
    def is_road_line(line):
        return True if line.__class__ == RoadLineType else False

    @staticmethod
    def is_yellow(line):
        return True if line in [
            RoadLineType.SOLID_DOUBLE_YELLOW, RoadLineType.PASSING_DOUBLE_YELLOW, RoadLineType.SOLID_SINGLE_YELLOW,
            RoadLineType.BROKEN_DOUBLE_YELLOW, RoadLineType.BROKEN_SINGLE_YELLOW
        ] else False

    @staticmethod
    def is_broken(line):
        return True if line in [
            RoadLineType.BROKEN_DOUBLE_YELLOW, RoadLineType.BROKEN_SINGLE_YELLOW, RoadLineType.BROKEN_SINGLE_WHITE
        ] else False


class RoadEdgeType(Enum):
    UNKNOWN = 0
    # Physical road boundary that doesn't have traffic on the other side (e.g., a curb or the k-rail on the right side of a freeway).
    BOUNDARY = 1
    # Physical road boundary that separates the car from other traffic (e.g. a k-rail or an island).
    MEDIAN = 2

    ENUM_TO_STR = {UNKNOWN: 'UNKNOWN', BOUNDARY: 'ROAD_EDGE_BOUNDARY', MEDIAN: 'ROAD_EDGE_MEDIAN'}

    @staticmethod
    def is_road_edge(edge):
        return True if edge.__class__ == RoadEdgeType else False

    @staticmethod
    def is_sidewalk(edge):
        return True if edge == RoadEdgeType.BOUNDARY else False


class AgentType(Enum):
    UNSET = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHER = 4

    ENUM_TO_STR = {UNSET: 'UNSET', VEHICLE: 'VEHICLE', PEDESTRIAN: 'PEDESTRIAN', CYCLIST: 'CYCLIST', OTHER: 'OTHER'}


class CustomUnpickler(pickle.Unpickler):
    def __init__(self, load_old_case, *args, **kwargs):
        super(CustomUnpickler, self).__init__(*args, **kwargs)
        self.load_old_case = load_old_case

    def find_class(self, module, name):
        if self.load_old_case:
            if name == 'AgentType':
                return AgentType
            elif name == "RoadLineType":
                return RoadLineType
            elif name == "RoadEdgeType":
                return RoadEdgeType
            return super().find_class(module, name)
        else:
            return super().find_class(module, name)


def parse_state(v_feature):
    state = v_feature["state"]
    ret = {}
    ret["position"] = v_feature["state"][..., 0:2]
    ret["size"] = v_feature["state"][..., 0:2]
    ret["heading"] = v_feature["state"][..., 0:2]
    ret["velocity"] = v_feature["state"][..., 0:2]
    ret["valid"] = v_feature["state"][..., 0:2]
    return ret


def convert_case(file_path, new_path):
    with open(file_path, "rb+") as file:
        data = pickle.load(file)
    new_data = {}
    new_data["id"] = data["id"]
    new_data["dynamic_map_states"] = [[{}]]  # old data has no traffic light info
    new_data["ts"] = data["ts"]  # old data has no traffic light info
    new_data["version"] = "old format"  # old data has no traffic light info
    new_data["sdc_track_index"] = str(data["sdc_index"])  # old data has no traffic light info
    new_data["map_features"] = data["map"]  # old data has no traffic light info
    new_track = {}
    for key, value in data["tracks"].items():
        new_track[str(key)] = value
    new_data["tracks"] = new_track

    # convert clas type
    for map_id, map_feature in new_data["map_features"].items():
        if "type" not in new_data["map_features"][map_id]:
            # filter sign and crosswalk
            continue
        if new_data["map_features"][map_id]["type"] == "center_lane":
            new_data["map_features"][map_id]["type"] = LaneTypeClass.LANE_SURFACE_STREET
        if isinstance(new_data["map_features"][map_id]["type"], Enum):
            new_data["map_features"][map_id]["type"] = new_data["map_features"][map_id]["type"].ENUM_TO_STR.value[
                new_data["map_features"][map_id]["type"].value]

    for v_id, v_feature in new_data["tracks"].items():
        new_v_feature = {}
        new_v_feature["type"] = v_feature["type"].ENUM_TO_STR.value[v_feature["type"].value]
        new_v_feature.update(parse_state(v_feature))
        new_data["tracks"][v_id] = new_v_feature
    with open(new_path, "wb+") as file:
        pickle.dump(new_path, file)


if __name__ == "__main__":
    file_path = AssetLoader.file_path("waymo", "0.pkl", return_raw_style=False)
    new_file_path = AssetLoader.file_path("waymo", "0_new.pkl", return_raw_style=False)
    convert_case(file_path, new_file_path)
