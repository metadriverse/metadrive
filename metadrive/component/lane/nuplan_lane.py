import logging

from metadrive.component.lane.point_lane import PointLane
from metadrive.constants import NuPlanLaneProperty
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.math_utils import norm
from metadrive.utils.waymo_utils.waymo_utils import read_waymo_data, convert_polyline_to_metadrive


class NuPlanLane(PointLane):
    def __init__(self, lane_id, lane_center_line_points, width, lane_meta_data):
        """
        Extract the lane information of one waymo lane, and do coordinate shift
        """
        super(NuPlanLane, self).__init__(lane_center_line_points, width)
        self.index = lane_id
        # TODO LQY, Remove it and fix the Edgeroadnetwork
        self.entry_lanes = lane_meta_data,
        self.exit_lanes = lane_meta_data,
        self.left_lanes = lane_meta_data,
        self.right_lanes = lane_meta_data

    def __del__(self):
        logging.debug("NuPlanLane is released")

    def destroy(self):
        self.index = None
        super(NuPlanLane, self).destroy()


if __name__ == "__main__":
    file_path = AssetLoader.file_path("waymo", "test.pkl", return_raw_style=False)
    data = read_waymo_data(file_path)
    print(data)
    lane = NuPlanLane(108, data["map"])
