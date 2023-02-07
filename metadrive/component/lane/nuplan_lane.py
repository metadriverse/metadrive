import logging
import math
from metadrive.constants import DrivableAreaProperty
import numpy as np

from metadrive.component.lane.point_lane import PointLane
from metadrive.utils.interpolating_line import InterpolatingLine
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.waymo_utils.waymo_utils import read_waymo_data


class NuPlanLane(PointLane):
    def __init__(self, lane_meta_data, nuplan_center):
        """
        Extract the lane information of one waymo lane, and do coordinate shift
        """
        super(NuPlanLane, self).__init__(self._extract_centerline(lane_meta_data, nuplan_center), None)
        self.index = lane_meta_data.id

        self.entry_lanes = lane_meta_data.incoming_edges,
        self.exit_lanes = lane_meta_data.outgoing_edges,
        self.left_lanes = lane_meta_data.adjacent_edges[0],
        self.right_lanes = lane_meta_data.adjacent_edges[-1]

        self.left_boundary = InterpolatingLine(self._get_boundary_points(lane_meta_data.left_boundary))
        self.right_boundary = InterpolatingLine(self._get_boundary_points(lane_meta_data.right_boundary))
        self.width = None

    @staticmethod
    def _extract_centerline(map_obj, nuplan_center):
        center = nuplan_center
        path = map_obj.baseline_path.discrete_path
        points = [np.array([pose.x - center[0], pose.y - center[1]]) for pose in path]
        return points

    def width_at(self, longitudinal: float) -> float:
        l_pos = self.left_boundary.position(longitudinal, 0)
        r_pos = self.right_boundary.position(longitudinal, 0)
        return min(np.linalg.norm(r_pos - l_pos), 6)

    def __del__(self):
        logging.debug("NuPlanLane is released")

    def destroy(self):
        self.index = None
        super(NuPlanLane, self).destroy()

    @staticmethod
    def _get_boundary_points(boundary):
        path = boundary.discrete_path
        points = [np.array([pose.x, pose.y]) for pose in path]
        return points

    def construct_lane_in_block(self, block, lane_index):
        segment_num = int(self.length / DrivableAreaProperty.LANE_SEGMENT_LENGTH)
        if segment_num == 0:
            middle = self.position(self.length / 2, 0)
            end = self.position(self.length, 0)
            theta = self.heading_theta_at(self.length / 2)
            width = self.width_at(0) + DrivableAreaProperty.SIDEWALK_LINE_DIST * 2
            self.construct_lane_segment(block, middle, width, self.length, theta, lane_index)

        for i in range(segment_num):
            middle = self.position(self.length * (i + .5) / segment_num, 0)
            end = self.position(self.length * (i + 1) / segment_num, 0)
            direction_v = end - middle
            theta = -math.atan2(direction_v[1], direction_v[0])
            width = self.width_at(self.length * (i + .5) / segment_num) + DrivableAreaProperty.SIDEWALK_LINE_DIST
            length = self.length
            self.construct_lane_segment(block, middle, width, length * 1.3 / segment_num, theta, lane_index)


if __name__ == "__main__":
    raise ValueError("Can not be run")
    file_path = AssetLoader.file_path("waymo", "test.pkl", return_raw_style=False)
    data = read_waymo_data(file_path)
    print(data)
    lane = NuPlanLane(108, data["map"])
