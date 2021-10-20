from metadrive.component.blocks.base_block import BaseBlock
from metadrive.constants import LineType, LineColor
from metadrive.constants import DrivableAreaProperty
from metadrive.utils.interpolating_line import InterpolatingLine
from metadrive.component.lane.waymo_lane import WaymoLane
from metadrive.constants import WaymoLaneProperty
from metadrive.component.road.road import Road


class WaymoBlock(BaseBlock):

    def __init__(self, block_index: int, global_network, random_seed, waymo_map_data: dict):
        self.waymo_map_data = waymo_map_data
        super(WaymoBlock, self).__init__(block_index, global_network, random_seed)

    def _sample_topology(self) -> bool:
        waymo_lanes = []
        for lane_id, data in self.waymo_map_data.items():
            if data.get("type", False) == WaymoLaneProperty.LANE_TYPE:
                if len(data[WaymoLaneProperty.POLYLINE]) <= 1:
                    continue
                waymo_lane = WaymoLane(lane_id, self.waymo_map_data)
                waymo_lanes.append(waymo_lane)
        self.block_network.add_road(Road("test", "test"), waymo_lanes)
        return True

    def create_in_world(self):
        """
        The lane line should be created separately
        """
        super(WaymoBlock, self).create_in_world()
        for lane_id, data in self.waymo_map_data.items():
            if data.get("type", False) in [WaymoLaneProperty.LANE_LINE_TYPE, WaymoLaneProperty.LANE_EDGE_TYPE]:
                if len(data[WaymoLaneProperty.POLYLINE]) <= 1:
                    continue
                self.construct_continuous_waymo_line([p[:-1] for p in data[WaymoLaneProperty.POLYLINE]])

    def construct_continuous_waymo_line(self, polyline):
        line = InterpolatingLine(polyline)
        segment_num = int(line.length / DrivableAreaProperty.LANE_SEGMENT_LENGTH)
        for segment in range(segment_num):
            start = line.get_point(DrivableAreaProperty.LANE_SEGMENT_LENGTH * segment)
            if segment == segment_num - 1:
                end = line.get_point(line.length)
            else:
                end = line.get_point((segment + 1) * DrivableAreaProperty.LANE_SEGMENT_LENGTH)
            WaymoLane.construct_lane_line_segment(self, start, end, LineColor.GREY, LineType.CONTINUOUS)

    def construct_broken_waymo_line(self, polyline):
        line = InterpolatingLine(polyline)
        segment_num = int(line.length / (2 * DrivableAreaProperty.STRIPE_LENGTH))
        for segment in range(segment_num):
            start = line.get_point(segment * DrivableAreaProperty.STRIPE_LENGTH * 2)
            end = line.get_point(
                segment * DrivableAreaProperty.STRIPE_LENGTH * 2 + DrivableAreaProperty.STRIPE_LENGTH)
            if segment == segment_num - 1:
                end = line.get_point(line.length - DrivableAreaProperty.STRIPE_LENGTH)
            WaymoLane.construct_lane_line_segment(self, start, end, LineColor.GREY, LineType.BROKEN)
