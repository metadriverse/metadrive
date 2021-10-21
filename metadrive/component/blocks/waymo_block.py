from metadrive.component.blocks.base_block import BaseBlock
from metadrive.component.lane.waymo_lane import WaymoLane
from metadrive.component.road.road import Road
from metadrive.constants import DrivableAreaProperty
from metadrive.constants import LineType, LineColor
from metadrive.constants import WaymoLaneProperty
from metadrive.utils.interpolating_line import InterpolatingLine
from metadrive.utils.waymo_map_utils import RoadLineType, RoadEdgeType


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
            type = data.get("type", None)
            if RoadLineType.is_road_line(type):
                if len(data[WaymoLaneProperty.POLYLINE]) <= 1:
                    continue
                if RoadLineType.is_broken(type):
                    self.construct_broken_waymo_line(
                        [p[:-1] for p in data[WaymoLaneProperty.POLYLINE]],
                        LineColor.YELLOW if RoadLineType.is_yellow(type) else LineColor.GREY
                    )
                else:
                    self.construct_continuous_waymo_line(
                        [p[:-1] for p in data[WaymoLaneProperty.POLYLINE]],
                        LineColor.YELLOW if RoadLineType.is_yellow(type) else LineColor.GREY
                    )
            elif RoadEdgeType.is_road_edge(type) and RoadEdgeType.is_sidewalk(type):
                self.construct_waymo_sidewalk([p[:-1] for p in data[WaymoLaneProperty.POLYLINE]])

    def construct_continuous_waymo_line(self, polyline, color):
        line = InterpolatingLine(polyline)
        segment_num = int(line.length / DrivableAreaProperty.STRIPE_LENGTH)
        for segment in range(segment_num):
            start = line.get_point(DrivableAreaProperty.STRIPE_LENGTH * segment)
            if segment == segment_num - 1:
                end = line.get_point(line.length)
            else:
                end = line.get_point((segment + 1) * DrivableAreaProperty.STRIPE_LENGTH)
            WaymoLane.construct_lane_line_segment(self, start, end, color, LineType.CONTINUOUS)

    def construct_broken_waymo_line(self, polyline, color):
        line = InterpolatingLine(polyline)
        segment_num = int(line.length / (2 * DrivableAreaProperty.STRIPE_LENGTH))
        for segment in range(segment_num):
            start = line.get_point(segment * DrivableAreaProperty.STRIPE_LENGTH * 2)
            end = line.get_point(segment * DrivableAreaProperty.STRIPE_LENGTH * 2 + DrivableAreaProperty.STRIPE_LENGTH)
            if segment == segment_num - 1:
                end = line.get_point(line.length - DrivableAreaProperty.STRIPE_LENGTH)
            WaymoLane.construct_lane_line_segment(self, start, end, color, LineType.BROKEN)

    def construct_waymo_sidewalk(self, polyline):
        line = InterpolatingLine(polyline)
        seg_len = 0.5
        segment_num = int(line.length / seg_len)
        for segment in range(segment_num):
            lane_start = line.get_point(segment * seg_len, -DrivableAreaProperty.SIDEWALK_WIDTH)
            if segment != segment_num - 1:
                lane_end = line.get_point((segment + 1) * seg_len, -DrivableAreaProperty.SIDEWALK_WIDTH)
            else:
                lane_end = line.get_point(line.length, -DrivableAreaProperty.SIDEWALK_WIDTH)
            WaymoLane.construct_sidewalk_segment(self, lane_start, lane_end, length=seg_len * 2)
