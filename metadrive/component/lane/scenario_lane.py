import logging
import math

from metadrive.component.lane.point_lane import PointLane
from metadrive.constants import DrivableAreaProperty, ScenarioLaneProperty
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.math_utils import norm, mph_to_kmh
from metadrive.scenario.utils import read_scenario_data, convert_polyline_to_metadrive


class ScenarioLane(PointLane):
    VIS_LANE_WIDTH = 6

    def __init__(self, lane_id: int, map_data: dict, need_lane_localization, coordinate_transform):
        """
        Extract the lane information of one lane, and do coordinate shift if required
        """
        self.need_lane_localization = need_lane_localization
        center_line_points = convert_polyline_to_metadrive(
            map_data[lane_id][ScenarioLaneProperty.POLYLINE], coordinate_transform=coordinate_transform
        )
        assert "speed_limit_kmh" in map_data[lane_id] or "speed_limit_mph" in map_data[lane_id]
        speed_limit_kmh = map_data[lane_id].get("speed_limit_kmh", None)
        if speed_limit_kmh is None:
            speed_limit_kmh = mph_to_kmh(map_data[lane_id]["speed_limit_mph"])
        super(ScenarioLane, self).__init__(
            center_line_points=center_line_points,
            width=self.get_lane_width(lane_id, map_data),
            speed_limit=speed_limit_kmh
        )
        self.index = lane_id
        self.lane_type = map_data[lane_id]["type"]
        self.entry_lanes = map_data[lane_id].get(ScenarioLaneProperty.ENTRY, None)
        self.exit_lanes = map_data[lane_id].get(ScenarioLaneProperty.EXIT, None)
        self.left_lanes = map_data[lane_id].get(ScenarioLaneProperty.LEFT_NEIGHBORS, None)
        self.right_lanes = map_data[lane_id].get(ScenarioLaneProperty.RIGHT_NEIGHBORS, None)

    def get_lane_width(self, lane_id, map_data):
        """
        We use this function to get possible lane width from raw data
        """
        if not (ScenarioLaneProperty.RIGHT_NEIGHBORS in map_data[lane_id]
                and ScenarioLaneProperty.LEFT_NEIGHBORS in map_data[lane_id]):
            return self.VIS_LANE_WIDTH
        right_lanes = map_data[lane_id][ScenarioLaneProperty.RIGHT_NEIGHBORS]
        left_lanes = map_data[lane_id][ScenarioLaneProperty.LEFT_NEIGHBORS]
        if len(right_lanes) + len(left_lanes) == 0:
            return max(sum(map_data[lane_id]["width"][0]), self.VIS_LANE_WIDTH)
        dist_to_left_lane = 0
        dist_to_right_lane = 0
        if len(right_lanes) > 0 and "feature_id" in right_lanes[0]:
            right_lane = map_data[right_lanes[0]["feature_id"]]
            self_start = int(right_lanes[0]["self_start_index"])
            neighbor_start = int(right_lanes[0]["neighbor_start_index"])
            n_point = right_lane[ScenarioLaneProperty.POLYLINE][neighbor_start]
            self_point = map_data[lane_id][ScenarioLaneProperty.POLYLINE][self_start]
            dist_to_right_lane = norm(n_point[0] - self_point[0], n_point[1] - self_point[1])
        if len(left_lanes) > 0 and "feature_id" in left_lanes[0]:
            left_lane = map_data[left_lanes[-1]["feature_id"]]
            self_start = int(left_lanes[-1]["self_start_index"])
            neighbor_start = int(left_lanes[-1]["neighbor_start_index"])
            n_point = left_lane[ScenarioLaneProperty.POLYLINE][neighbor_start]
            self_point = map_data[lane_id][ScenarioLaneProperty.POLYLINE][self_start]
            dist_to_left_lane = norm(n_point[0] - self_point[0], n_point[1] - self_point[1])
        return max(dist_to_left_lane, dist_to_right_lane, self.VIS_LANE_WIDTH)

    def __del__(self):
        logging.debug("ScenarioLane is released")

    def destroy(self):
        self.index = None
        self.entry_lanes = None
        self.exit_lanes = None
        self.left_lanes = None
        self.right_lanes = None
        super(ScenarioLane, self).destroy()

    def construct_lane_in_block(self, block, lane_index):
        """
        Modified from base class, the width is set to 6.5
        """
        # build physics contact
        if self.need_lane_localization:
            super(ScenarioLane, self).construct_lane_in_block(block, lane_index)
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
                theta = math.atan2(direction_v[1], direction_v[0])
                length = self.length
                self._construct_lane_only_vis_segment(
                    block, middle, self.VIS_LANE_WIDTH, length * 1.3 / segment_num, theta
                )


if __name__ == "__main__":
    file_path = AssetLoader.file_path("waymo", "test.pkl", return_raw_style=False)
    data = read_scenario_data(file_path)
    print(data)
    lane = ScenarioLane(108, data["map_features"], coordinate_transform=True)
