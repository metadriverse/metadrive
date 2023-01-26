import logging

from metadrive.component.lane.point_lane import PointLane
from metadrive.constants import NuPlanLaneProperty
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.math_utils import norm
from metadrive.utils.waymo_utils.waymo_utils import read_waymo_data, convert_polyline_to_metadrive


class NuPlanLane(PointLane):
    def __init__(self, waymo_lane_id: int, waymo_map_data: dict):
        """
        Extract the lane information of one waymo lane, and do coordinate shift
        """
        raise ValueError
        super(NuPlanLane, self).__init__(
            convert_polyline_to_metadrive(waymo_map_data[waymo_lane_id][NuPlanLaneProperty.POLYLINE]),
            self.get_lane_width(waymo_lane_id, waymo_map_data)
        )
        self.index = waymo_lane_id
        self.entry_lanes = waymo_map_data[waymo_lane_id][NuPlanLaneProperty.ENTRY]
        self.exit_lanes = waymo_map_data[waymo_lane_id][NuPlanLaneProperty.EXIT]
        self.left_lanes = waymo_map_data[waymo_lane_id][NuPlanLaneProperty.LEFT_NEIGHBORS]
        self.right_lanes = waymo_map_data[waymo_lane_id][NuPlanLaneProperty.RIGHT_NEIGHBORS]
        # left_type = LineType.CONTINUOUS if len(self.left_lanes) == 0 else LineType.NONE
        # righ_type = LineType.CONTINUOUS if len(self.right_lanes) == 0 else LineType.NONE
        # self.line_types = (left_type, righ_type)

    @staticmethod
    def get_lane_width(waymo_lane_id, waymo_map_data):
        """
        We use this function to get possible lane width from raw data
        """
        raise ValueError
        right_lanes = waymo_map_data[waymo_lane_id][NuPlanLaneProperty.RIGHT_NEIGHBORS]
        left_lanes = waymo_map_data[waymo_lane_id][NuPlanLaneProperty.LEFT_NEIGHBORS]
        if len(right_lanes) + len(left_lanes) == 0:
            return max(sum(waymo_map_data[waymo_lane_id]["width"][0]), 6)
        dist_to_left_lane = 0
        dist_to_right_lane = 0
        if len(right_lanes) > 0:
            right_lane = waymo_map_data[right_lanes[0]["id"]]
            self_start = right_lanes[0]["indexes"][0]
            neighbor_start = right_lanes[0]["indexes"][2]
            n_point = right_lane[NuPlanLaneProperty.POLYLINE][neighbor_start]
            self_point = waymo_map_data[waymo_lane_id][NuPlanLaneProperty.POLYLINE][self_start]
            dist_to_right_lane = norm(n_point[0] - self_point[0], n_point[1] - self_point[1])
        if len(left_lanes) > 0:
            left_lane = waymo_map_data[left_lanes[-1]["id"]]
            self_start = left_lanes[-1]["indexes"][0]
            neighbor_start = left_lanes[-1]["indexes"][2]
            n_point = left_lane[NuPlanLaneProperty.POLYLINE][neighbor_start]
            self_point = waymo_map_data[waymo_lane_id][NuPlanLaneProperty.POLYLINE][self_start]
            dist_to_left_lane = norm(n_point[0] - self_point[0], n_point[1] - self_point[1])
        return max(dist_to_left_lane, dist_to_right_lane, 6)

    def __del__(self):
        logging.debug("NuPlanLane is released")

    def destroy(self):
        self.index = None
        self.entry_lanes = None
        self.exit_lanes = None
        self.left_lanes = None
        self.right_lanes = None
        super(NuPlanLane, self).destroy()


if __name__ == "__main__":
    file_path = AssetLoader.file_path("waymo", "test.pkl", return_raw_style=False)
    data = read_waymo_data(file_path)
    print(data)
    lane = NuPlanLane(108, data["map"])
