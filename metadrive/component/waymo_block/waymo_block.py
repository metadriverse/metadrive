import math
from metadrive.utils.coordinates_shift import panda_heading

from metadrive.component.block.base_block import BaseBlock
from metadrive.component.lane.waymo_lane import WaymoLane
from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork
from metadrive.constants import DrivableAreaProperty
from metadrive.constants import LineType, LineColor
from metadrive.utils.waymo_utils.waymo_type import WaymoLaneType, WaymoLaneProperty
from metadrive.engine.engine_utils import get_engine
from metadrive.utils.interpolating_line import InterpolatingLine
from metadrive.utils.math_utils import wrap_to_pi, norm
from metadrive.utils.waymo_utils.utils import convert_polyline_to_metadrive
from metadrive.utils.waymo_utils.waymo_type import WaymoRoadLineType, WaymoRoadEdgeType


class WaymoBlock(BaseBlock):
    def __init__(self, block_index: int, global_network, random_seed, map_index, need_lane_localization):
        # self.waymo_map_data = waymo_map_data
        self.need_lane_localization = need_lane_localization
        self.map_index = map_index
        super(WaymoBlock, self).__init__(block_index, global_network, random_seed)

    @property
    def waymo_map_data(self):
        e = get_engine()
        return e.data_manager.get_scenario(self.map_index, should_copy=False)["map_features"]

    def _sample_topology(self) -> bool:
        for lane_id, data in self.waymo_map_data.items():
            if WaymoLaneType.is_lane(data.get("type", False)):
                if len(data[WaymoLaneProperty.POLYLINE]) <= 1:
                    continue
                waymo_lane = WaymoLane(
                    lane_id,
                    self.waymo_map_data,
                    self.need_lane_localization,
                    coordinate_transform=self.coordinate_transform
                )
                self.block_network.add_lane(waymo_lane)
        return True

    def create_in_world(self):
        """
        The lane line should be created separately
        """
        graph = self.block_network.graph
        for id, lane_info in graph.items():
            lane = lane_info.lane
            lane.construct_lane_in_block(self, lane_index=id)
            # lane.construct_lane_line_in_block(self, [True if len(lane.left_lanes) == 0 else False,
            #                                          True if len(lane.right_lanes) == 0 else False, ])
        # draw
        for lane_id, data in self.waymo_map_data.items():
            type = data.get("type", None)
            if WaymoRoadLineType.is_road_line(type):
                if len(data[WaymoLaneProperty.POLYLINE]) <= 1:
                    continue
                if WaymoRoadLineType.is_broken(type):
                    self.construct_waymo_broken_line(
                        convert_polyline_to_metadrive(
                            data[WaymoLaneProperty.POLYLINE], coordinate_transform=self.coordinate_transform
                        ), LineColor.YELLOW if WaymoRoadLineType.is_yellow(type) else LineColor.GREY
                    )
                else:
                    self.construct_waymo_continuous_line(
                        convert_polyline_to_metadrive(
                            data[WaymoLaneProperty.POLYLINE], coordinate_transform=self.coordinate_transform
                        ), LineColor.YELLOW if WaymoRoadLineType.is_yellow(type) else LineColor.GREY
                    )
            # elif WaymoRoadEdgeType.is_road_edge(type) and WaymoRoadEdgeType.is_sidewalk(type):
            #     self.construct_waymo_sidewalk(
            #         convert_polyline_to_metadrive(
            #             data[WaymoLaneProperty.POLYLINE], coordinate_transform=self.coordinate_transform
            #         )
            #     )
            # elif WaymoRoadEdgeType.is_road_edge(type) and not WaymoRoadEdgeType.is_sidewalk(type):
            #     self.construct_waymo_continuous_line(
            #         convert_polyline_to_metadrive(
            #             data[WaymoLaneProperty.POLYLINE], coordinate_transform=self.coordinate_transform
            #         ), LineColor.GREY
            #     )
            # else:
            #     raise ValueError("Can not build lane line type: {}".format(type))
            elif WaymoRoadEdgeType.is_road_edge(type):
                self.construct_waymo_sidewalk(
                    convert_polyline_to_metadrive(
                        data[WaymoLaneProperty.POLYLINE], coordinate_transform=self.coordinate_transform
                    )
                )

    def construct_waymo_continuous_line(self, polyline, color):
        line = InterpolatingLine(polyline)
        segment_num = int(line.length / DrivableAreaProperty.STRIPE_LENGTH)
        for segment in range(segment_num):
            start = line.get_point(DrivableAreaProperty.STRIPE_LENGTH * segment)
            if segment == segment_num - 1:
                end = line.get_point(line.length)
            else:
                end = line.get_point((segment + 1) * DrivableAreaProperty.STRIPE_LENGTH)
            node_path_list = WaymoLane.construct_lane_line_segment(self, start, end, color, LineType.CONTINUOUS)
            self._node_path_list.extend(node_path_list)

    def construct_waymo_broken_line(self, polyline, color):
        line = InterpolatingLine(polyline)
        segment_num = int(line.length / (2 * DrivableAreaProperty.STRIPE_LENGTH))
        for segment in range(segment_num):
            start = line.get_point(segment * DrivableAreaProperty.STRIPE_LENGTH * 2)
            end = line.get_point(segment * DrivableAreaProperty.STRIPE_LENGTH * 2 + DrivableAreaProperty.STRIPE_LENGTH)
            if segment == segment_num - 1:
                end = line.get_point(line.length - DrivableAreaProperty.STRIPE_LENGTH)
            node_path_list = WaymoLane.construct_lane_line_segment(self, start, end, color, LineType.BROKEN)
            self._node_path_list.extend(node_path_list)

    def construct_waymo_sidewalk(self, polyline):
        line = InterpolatingLine(polyline)
        seg_len = DrivableAreaProperty.LANE_SEGMENT_LENGTH
        segment_num = int(line.length / seg_len)
        # last_theta = None
        for segment in range(segment_num):
            lane_start = line.get_point(segment * seg_len)
            lane_end = line.get_point((segment + 1) * seg_len)
            if segment == segment_num - 1:
                lane_end = line.get_point(line.length)
            direction_v = lane_end - lane_start
            theta = panda_heading(math.atan2(direction_v[1], direction_v[0]))
            # if segment == segment_num - 1:
            #     factor = 1
            # else:
            #     factor = 1.25
            #     if last_theta is not None:
            #         diff = wrap_to_pi(theta) - wrap_to_pi(last_theta)
            #         if diff > 0:
            #             factor += math.sin(abs(diff) / 2) * DrivableAreaProperty.SIDEWALK_WIDTH / norm(
            #                 lane_start[0] - lane_end[0], lane_start[1] - lane_end[1]
            #             ) + 0.15
            #         else:
            #             factor -= math.sin(abs(diff) / 2) * DrivableAreaProperty.SIDEWALK_WIDTH / norm(
            #                 lane_start[0] - lane_end[0], lane_start[1] - lane_end[1]
            #             )
            last_theta = theta
            node_path_list = WaymoLane.construct_sidewalk_segment(
                self, lane_start, lane_end, length_multiply=1, extra_thrust=0, width=0.2
            )
            self._node_path_list.extend(node_path_list)

    @property
    def block_network_type(self):
        return EdgeRoadNetwork

    def destroy(self):
        self.map_index = None
        # self.waymo_map_data = None
        super(WaymoBlock, self).destroy()

    def __del__(self):
        self.destroy()
        super(WaymoBlock, self).__del__()
        # print("Waymo Block is being deleted.")

    @property
    def coordinate_transform(self):
        return self.engine.global_config["coordinate_transform"]

    # @property
    # def waymo_map_data(self):
    #     e = get_engine()
    #     return e.data_manager.get_scenario(self.map_index)["map"]
