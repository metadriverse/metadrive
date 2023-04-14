import math

import numpy as np

from metadrive.component.block.base_block import BaseBlock
from metadrive.component.lane.scenario_lane import ScenarioLane
from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork
from metadrive.constants import DrivableAreaProperty
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.constants import PGLineType, PGLineColor
from metadrive.engine.engine_utils import get_engine
from metadrive.type import MetaDriveType
from metadrive.utils.coordinates_shift import panda_heading
from metadrive.utils.interpolating_line import InterpolatingLine


class ScenarioBlock(BaseBlock):
    def __init__(self, block_index: int, global_network, random_seed, map_index, need_lane_localization):
        # self.map_data = map_data
        self.need_lane_localization = need_lane_localization
        self.map_index = map_index
        super(ScenarioBlock, self).__init__(block_index, global_network, random_seed)

    @property
    def map_data(self):
        e = get_engine()
        return e.data_manager.get_scenario(self.map_index, should_copy=False)["map_features"]

    def _sample_topology(self) -> bool:
        for lane_id, data in self.map_data.items():
            if MetaDriveType.is_lane(data.get("type", False)):
                if len(data[ScenarioDescription.POLYLINE]) <= 1:
                    continue
                lane = ScenarioLane(lane_id, self.map_data, self.need_lane_localization)
                self.block_network.add_lane(lane)
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
        for lane_id, data in self.map_data.items():
            type = data.get("type", None)
            if MetaDriveType.is_road_line(type):
                if len(data[ScenarioDescription.POLYLINE]) <= 1:
                    continue
                if MetaDriveType.is_broken_line(type):
                    self.construct_broken_line(
                        np.asarray(data[ScenarioDescription.POLYLINE]),
                        PGLineColor.YELLOW if MetaDriveType.is_yellow_line(type) else PGLineColor.GREY
                    )
                else:
                    self.construct_continuous_line(
                        np.asarray(data[ScenarioDescription.POLYLINE]),
                        PGLineColor.YELLOW if MetaDriveType.is_yellow_line(type) else PGLineColor.GREY
                    )
            # elif MetaDriveType.is_road_edge(type) and MetaDriveType.is_sidewalk(type):
            #     self.construct_sidewalk(
            #         convert_polyline_to_metadrive(
            #             data[ScenarioDescription.POLYLINE], coordinate_transform=self.coordinate_transform
            #         )
            #     )
            # elif MetaDriveType.is_road_edge(type) and not MetaDriveType.is_sidewalk(type):
            #     self.construct_continuous_line(
            #         convert_polyline_to_metadrive(
            #             data[ScenarioDescription.POLYLINE], coordinate_transform=self.coordinate_transform
            #         ), PGLineColor.GREY
            #     )
            # else:
            #     raise ValueError("Can not build lane line type: {}".format(type))
            elif MetaDriveType.is_road_edge(type):
                self.construct_sidewalk(np.asarray(data[ScenarioDescription.POLYLINE]))

    def construct_continuous_line(self, polyline, color):
        line = InterpolatingLine(polyline)
        segment_num = int(line.length / DrivableAreaProperty.STRIPE_LENGTH)
        for segment in range(segment_num):
            start = line.get_point(DrivableAreaProperty.STRIPE_LENGTH * segment)
            if segment == segment_num - 1:
                end = line.get_point(line.length)
            else:
                end = line.get_point((segment + 1) * DrivableAreaProperty.STRIPE_LENGTH)
            node_path_list = ScenarioLane.construct_lane_line_segment(self, start, end, color, PGLineType.CONTINUOUS)
            self._node_path_list.extend(node_path_list)

    def construct_broken_line(self, polyline, color):
        line = InterpolatingLine(polyline)
        segment_num = int(line.length / (2 * DrivableAreaProperty.STRIPE_LENGTH))
        for segment in range(segment_num):
            start = line.get_point(segment * DrivableAreaProperty.STRIPE_LENGTH * 2)
            end = line.get_point(segment * DrivableAreaProperty.STRIPE_LENGTH * 2 + DrivableAreaProperty.STRIPE_LENGTH)
            if segment == segment_num - 1:
                end = line.get_point(line.length - DrivableAreaProperty.STRIPE_LENGTH)
            node_path_list = ScenarioLane.construct_lane_line_segment(self, start, end, color, PGLineType.BROKEN)
            self._node_path_list.extend(node_path_list)

    def construct_sidewalk(self, polyline):
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
            node_path_list = ScenarioLane.construct_sidewalk_segment(
                self, lane_start, lane_end, length_multiply=1, extra_thrust=0, width=0.2
            )
            self._node_path_list.extend(node_path_list)

    @property
    def block_network_type(self):
        return EdgeRoadNetwork

    def destroy(self):
        self.map_index = None
        # self.map_data = None
        super(ScenarioBlock, self).destroy()

    def __del__(self):
        self.destroy()
        super(ScenarioBlock, self).__del__()
