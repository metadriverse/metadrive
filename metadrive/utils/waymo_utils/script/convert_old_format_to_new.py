"""
I only use this script to convert 3 old pickle cases into new format. People should generate the dara by using
convert_waymo_to_metadrive.py
"""
import pickle
import sys
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
    def find_class(self, module, name):
        if name == 'AgentType':
            return AgentType
        elif name == "RoadLineType":
            return RoadLineType
        elif name == "RoadEdgeType":
            return RoadEdgeType
        return super(CustomUnpickler, self).find_class(module, name)


def convert_case(file_path, new_path):
    with open(file_path, "rb+") as file:
        loader = CustomUnpickler(file)
        data = loader.load()
    new_data = {}
    new_data["id"] = data["id"]
    new_data["dynamic_map_states"] = {}  # old data has no traffic light info
    new_data["ts"] = data["ts"]  # old data has no traffic light info
    new_data["version"] = "2022-10"  # old data has no traffic light info
    new_data["sdc_track_index"] = str(data["sdc_index"])  # old data has no traffic light info
    new_track = {}
    for key, value in data["tracks"].items():
        new_track[str(key)] = value
    new_data["tracks"] = new_track

    # convert clas type
    new_data["map_features"] = {}
    for map_id, map_feature in data["map"].items():
        if "type" not in data["map"][map_id]:
            # filter sign and crosswalk
            continue
        new_data["map_features"][map_id] = data["map"][map_id]
        if new_data["map_features"][map_id]["type"] == "center_lane":
            # remove neighbor and boundary
            new_data["map_features"][map_id]["type"] = "LANE_SURFACE_STREET"
            for b in new_data["map_features"][map_id]["left_boundaries"]:
                b["type"] = b["type"].ENUM_TO_STR.value[b["type"].value]
            for b in new_data["map_features"][map_id]["right_boundaries"]:
                b["type"] = b["type"].ENUM_TO_STR.value[b["type"].value]

            new_neighbor = []
            for b in new_data["map_features"][map_id]["right_neighbor"]:
                new_neighbor.append({"id": b["id"]})
            new_data["map_features"][map_id]["right_neighbor"] = new_neighbor
            new_neighbor = []

            for b in new_data["map_features"][map_id]["left_boundaries"]:
                new_neighbor.append({"id": b["id"]})
            new_data["map_features"][map_id]["left_neighbor"] = new_neighbor
            new_data["map_features"][map_id]["entry_lanes"] = new_data["map_features"][map_id]["entry"]
            new_data["map_features"][map_id]["exit_lanes"] = new_data["map_features"][map_id]["exit"]
        if isinstance(new_data["map_features"][map_id]["type"], Enum):
            new_data["map_features"][map_id]["type"] = new_data["map_features"][map_id]["type"].ENUM_TO_STR.value[
                new_data["map_features"][map_id]["type"].value]

    for v_id, v_feature in new_data["tracks"].items():
        new_v_feature = {}
        new_v_feature["type"] = v_feature["type"].ENUM_TO_STR.value[v_feature["type"].value]

        state_dict = {}
        state_dict["position"] = v_feature["state"][..., 0:3]
        state_dict["size"] = v_feature["state"][..., 3:6]
        state_dict["heading"] = v_feature["state"][..., 6:7]
        state_dict["velocity"] = v_feature["state"][..., 7:9]
        state_dict["valid"] = v_feature["state"][..., 9:10]

        new_v_feature["state"] = state_dict

        new_v_feature["metadata"] = dict(
            track_length=state_dict["position"].shape[0],
            type=new_v_feature["type"],
            object_id=v_id
        )

        new_data["tracks"][v_id] = new_v_feature
    with open(new_path, "wb+") as file:
        pickle.dump(new_data, file)


if __name__ == "__main__":
    for i in range(3):
        file_path = AssetLoader.file_path("waymo", "{}.pkl".format(i), return_raw_style=False)
        new_file_path = AssetLoader.file_path("waymo", "{}.pkl".format(i + 3), return_raw_style=False)
        convert_case(file_path, new_file_path)
    sys.exit()
