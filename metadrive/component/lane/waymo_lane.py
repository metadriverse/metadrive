from metadrive.component.lane.waypoint_lane import WayPointLane
from metadrive.utils.waymo_map_utils import read_waymo_data
from metadrive.engine.asset_loader import AssetLoader
from metadrive.constants import WaymoLaneProperty


class WaymoLane(WayPointLane):

    def __init__(self, waymo_lane_id: int, waymo_map_data: dict):
        """
        Extract the lane information of one waymo lane
        """
        super(WaymoLane, self).__init__(
            [p[:-1] for p in waymo_map_data[waymo_lane_id][WaymoLaneProperty.CENTER_POINTS]], 4)
        self.index = waymo_lane_id

    def get_lane_width(self, waymo_lane_id, waymo_map_data):
        """
        We use this function to get possible lane width from raw data
        """
        # right_ns = waymo_map_data[waymo_lane_id][WaymoLaneProperty.RIGHT_NEIGHBORS]
        # right_n = right_ns[0] if len(right_ns) > 0 else None
        # left_ns = waymo_map_data[waymo_lane_id][WaymoLaneProperty.LEFT_NEIGHBORS]
        # left_n = left_ns[-1] if len(left_ns) > 0 else None
        # if right_n is not None:
        #
        # elif left_n is not None
        return 4


if __name__ == "__main__":
    file_path = AssetLoader.file_path("waymo", "test.pkl", linux_style=False)
    data = read_waymo_data(file_path)
    print(data)
    lane = WaymoLane(108, data["map"])
