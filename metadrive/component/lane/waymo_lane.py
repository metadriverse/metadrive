import logging

from metadrive.component.lane.waypoint_lane import WayPointLane, LineType
from metadrive.utils.math_utils import norm
from metadrive.constants import WaymoLaneProperty
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.waymo_utils.waymo_utils import read_waymo_data, convert_polyline_to_metadrive


class WaymoLane(WayPointLane):
    def __init__(self, waymo_lane_id: int, waymo_map_data: dict):
        """
        Extract the lane information of one waymo lane, and do coordinate shift
        """
        super(WaymoLane, self).__init__(
            convert_polyline_to_metadrive(waymo_map_data[waymo_lane_id][WaymoLaneProperty.POLYLINE]),
            self.get_lane_width(waymo_lane_id, waymo_map_data)
        )
        self.index = waymo_lane_id
        self.entry_lanes = waymo_map_data[waymo_lane_id][WaymoLaneProperty.ENTRY]
        self.exit_lanes = waymo_map_data[waymo_lane_id][WaymoLaneProperty.EXIT]
        self.left_lanes = waymo_map_data[waymo_lane_id][WaymoLaneProperty.LEFT_NEIGHBORS]
        self.right_lanes = waymo_map_data[waymo_lane_id][WaymoLaneProperty.RIGHT_NEIGHBORS]
        # left_type = LineType.CONTINUOUS if len(self.left_lanes) == 0 else LineType.NONE
        # righ_type = LineType.CONTINUOUS if len(self.right_lanes) == 0 else LineType.NONE
        # self.line_types = (left_type, righ_type)

    @staticmethod
    def get_lane_width(waymo_lane_id, waymo_map_data):
        """
        We use this function to get possible lane width from raw data
        """
        right_lanes = waymo_map_data[waymo_lane_id][WaymoLaneProperty.RIGHT_NEIGHBORS]
        left_lanes = waymo_map_data[waymo_lane_id][WaymoLaneProperty.LEFT_NEIGHBORS]
        if len(right_lanes) + len(left_lanes) == 0:
            return max(sum(waymo_map_data[waymo_lane_id]["width"][0]), 6)
        dist_to_left_lane = 0
        dist_to_right_lane = 0
        if len(right_lanes) > 0:
            right_lane = waymo_map_data[right_lanes[0]["id"]]
            self_start = right_lanes[0]["indexes"][0]
            neighbor_start = right_lanes[0]["indexes"][2]
            n_point = right_lane[WaymoLaneProperty.POLYLINE][neighbor_start]
            self_point = waymo_map_data[waymo_lane_id][WaymoLaneProperty.POLYLINE][self_start]
            dist_to_right_lane = norm(n_point[0] - self_point[0], n_point[1] - self_point[1])
        if len(left_lanes) > 0:
            left_lane = waymo_map_data[left_lanes[-1]["id"]]
            self_start = left_lanes[-1]["indexes"][0]
            neighbor_start = left_lanes[-1]["indexes"][2]
            n_point = left_lane[WaymoLaneProperty.POLYLINE][neighbor_start]
            self_point = waymo_map_data[waymo_lane_id][WaymoLaneProperty.POLYLINE][self_start]
            dist_to_left_lane = norm(n_point[0] - self_point[0], n_point[1] - self_point[1])
        return max(dist_to_left_lane, dist_to_right_lane, 6)

    def __del__(self):
        logging.debug("WaymoLane is released")


if __name__ == "__main__":
    file_path = AssetLoader.file_path("waymo", "test.pkl", return_raw_style=False)
    data = read_waymo_data(file_path)
    print(data)
    lane = WaymoLane(108, data["map"])
