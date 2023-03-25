import logging
import math

from metadrive.component.lane.point_lane import PointLane
from metadrive.constants import DrivableAreaProperty
from metadrive.utils.waymo_utils.waymo_type import WaymoLaneProperty
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.math_utils import norm
from metadrive.utils.waymo_utils.utils import read_waymo_data, convert_polyline_to_metadrive


class WaymoLane(PointLane):
    VIS_LANE_WIDTH = 6

    def __init__(self, waymo_lane_id: int, waymo_map_data: dict, need_lane_localization):
        """
        Extract the lane information of one waymo lane, and do coordinate shift
        """
        self.need_lane_localization = need_lane_localization
        super(WaymoLane, self).__init__(
            convert_polyline_to_metadrive(waymo_map_data[waymo_lane_id][WaymoLaneProperty.POLYLINE]),
            self.get_lane_width(waymo_lane_id, waymo_map_data),
            speed_limit=waymo_map_data[waymo_lane_id].get("speed_limit_mph", 0) * 1.609344  # to km/h
        )
        self.index = waymo_lane_id
        self.lane_type = waymo_map_data[waymo_lane_id]["type"]
        self.entry_lanes = waymo_map_data[waymo_lane_id][WaymoLaneProperty.ENTRY]
        self.exit_lanes = waymo_map_data[waymo_lane_id][WaymoLaneProperty.EXIT]
        self.left_lanes = waymo_map_data[waymo_lane_id][WaymoLaneProperty.LEFT_NEIGHBORS]
        self.right_lanes = waymo_map_data[waymo_lane_id][WaymoLaneProperty.RIGHT_NEIGHBORS]

    def get_lane_width(self, waymo_lane_id, waymo_map_data):
        """
        We use this function to get possible lane width from raw data
        """
        right_lanes = waymo_map_data[waymo_lane_id][WaymoLaneProperty.RIGHT_NEIGHBORS]
        left_lanes = waymo_map_data[waymo_lane_id][WaymoLaneProperty.LEFT_NEIGHBORS]
        if len(right_lanes) + len(left_lanes) == 0:
            return max(sum(waymo_map_data[waymo_lane_id]["width"][0]), self.VIS_LANE_WIDTH)
        dist_to_left_lane = 0
        dist_to_right_lane = 0
        if len(right_lanes) > 0 and "feature_id" in right_lanes[0]:
            right_lane = waymo_map_data[right_lanes[0]["feature_id"]]
            self_start = right_lanes[0]["self_start_index"]
            neighbor_start = right_lanes[0]["neighbor_start_index"]
            n_point = right_lane[WaymoLaneProperty.POLYLINE][neighbor_start]
            self_point = waymo_map_data[waymo_lane_id][WaymoLaneProperty.POLYLINE][self_start]
            dist_to_right_lane = norm(n_point[0] - self_point[0], n_point[1] - self_point[1])
        if len(left_lanes) > 0 and "feature_id" in left_lanes[0]:
            left_lane = waymo_map_data[left_lanes[-1]["feature_id"]]
            self_start = left_lanes[-1]["self_start_index"]
            neighbor_start = left_lanes[-1]["neighbor_start_index"]
            n_point = left_lane[WaymoLaneProperty.POLYLINE][neighbor_start]
            self_point = waymo_map_data[waymo_lane_id][WaymoLaneProperty.POLYLINE][self_start]
            dist_to_left_lane = norm(n_point[0] - self_point[0], n_point[1] - self_point[1])
        return max(dist_to_left_lane, dist_to_right_lane, self.VIS_LANE_WIDTH)

    def __del__(self):
        logging.debug("WaymoLane is released")

    def destroy(self):
        self.index = None
        self.entry_lanes = None
        self.exit_lanes = None
        self.left_lanes = None
        self.right_lanes = None
        super(WaymoLane, self).destroy()

    def construct_lane_in_block(self, block, lane_index):
        """
        Modified from base class, the width is set to 6.5
        """
        # build physics contact
        if self.need_lane_localization:
            super(WaymoLane, self).construct_lane_in_block(block, lane_index)
        else:
            lane = self
            if lane_index is not None:
                lane.index = lane_index

            # build visualization
            segment_num = int(self.length / DrivableAreaProperty.LANE_SEGMENT_LENGTH)
            if segment_num == 0:
                middle = self.position(self.length / 2, 0)
                end = self.position(self.length, 0)
                theta = self.heading_theta_at(self.length / 2)
                self._construct_lane_only_vis_segment(block, middle, self.VIS_LANE_WIDTH, self.length, theta)

            for i in range(segment_num):
                middle = self.position(self.length * (i + .5) / segment_num, 0)
                end = self.position(self.length * (i + 1) / segment_num, 0)
                direction_v = end - middle
                theta = -math.atan2(direction_v[1], direction_v[0])
                length = self.length
                self._construct_lane_only_vis_segment(
                    block, middle, self.VIS_LANE_WIDTH, length * 1.3 / segment_num, theta
                )


if __name__ == "__main__":
    file_path = AssetLoader.file_path("waymo", "test.pkl", return_raw_style=False)
    data = read_waymo_data(file_path)
    print(data)
    lane = WaymoLane(108, data["map_features"])
