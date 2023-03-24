"""
I only use this script to convert 3 old pickle cases into new format. People should generate the dara by using
convert_waymo_to_metadrive.py
"""
import pickle
from enum import Enum

from metadrive.engine.asset_loader import AssetLoader


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


def convert_case(file_path, new_path):
    with open(file_path, "rb+") as file:
        data = pickle.load(file)
    new_data = {}
    new_data["id"] = data["id"]
    new_data["dynamic_map_states"] = [[{}]]  # old data has no traffic light info
    new_data["ts"] = data["ts"]  # old data has no traffic light info
    new_data["version"] = "old format"  # old data has no traffic light info
    new_data["sdc_track_index"] = data["sdc_index"]  # old data has no traffic light info
    new_data["map_features"] = data["map"]  # old data has no traffic light info
    new_data["tracks"] = data["tracks"]  # old data has no traffic light info

    with open(new_path, "wb+") as file:
        pickle.dump(new_path, file)


if __name__ == "__main__":
    file_path = AssetLoader.file_path("waymo", "0.pkl", return_raw_style=False)
    new_file_path = AssetLoader.file_path("waymo", "0_new.pkl", return_raw_style=False)
    convert_case(file_path, new_file_path)
