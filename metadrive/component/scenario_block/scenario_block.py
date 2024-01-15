import numpy as np

from metadrive.component.block.base_block import BaseBlock
from metadrive.component.lane.scenario_lane import ScenarioLane
from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork
from metadrive.constants import PGDrivableAreaProperty
from metadrive.constants import PGLineType, PGLineColor
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.type import MetaDriveType
from metadrive.utils.interpolating_line import InterpolatingLine
from metadrive.utils.math import norm


class ScenarioBlock(BaseBlock):
    LINE_CULL_DIST = 500

    def __init__(self, block_index: int, global_network, random_seed, map_index, map_data, need_lane_localization):
        # self.map_data = map_data
        self.need_lane_localization = need_lane_localization
        self.map_index = map_index
        self.map_data = map_data
        super(ScenarioBlock, self).__init__(block_index, global_network, random_seed)

    def _sample_topology(self) -> bool:
        for object_id, data in self.map_data.items():
            if MetaDriveType.is_lane(data.get("type", False)):
                if len(data[ScenarioDescription.POLYLINE]) <= 1:
                    continue
                lane = ScenarioLane(object_id, self.map_data, self.need_lane_localization)
                self.block_network.add_lane(lane)
            elif MetaDriveType.is_sidewalk(data["type"]):
                self.sidewalks[object_id] = {
                    ScenarioDescription.TYPE: MetaDriveType.BOUNDARY_SIDEWALK,
                    ScenarioDescription.POLYGON: data[ScenarioDescription.POLYGON]
                }
            elif MetaDriveType.is_crosswalk(data["type"]):
                self.crosswalks[object_id] = {
                    ScenarioDescription.TYPE: MetaDriveType.CROSSWALK,
                    ScenarioDescription.POLYGON: data[ScenarioDescription.POLYGON]
                }
            else:
                pass
        return True

    def create_in_world(self):
        """
        The lane line should be created separately
        """
        graph = self.block_network.graph
        for id, lane_info in graph.items():
            lane = lane_info.lane
            self._construct_lane(lane, lane_index=id)
        # draw
        for lane_id, data in self.map_data.items():
            type = data.get("type", None)
            if ScenarioDescription.POLYLINE in data and len(data[ScenarioDescription.POLYLINE]) <= 1:
                continue
            if MetaDriveType.is_road_line(type):
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
            elif MetaDriveType.is_road_boundary_line(type):
                self.construct_continuous_line(np.asarray(data[ScenarioDescription.POLYLINE]), color=PGLineColor.GREY)
        self._construct_sidewalk()
        self._construct_crosswalk()

    def construct_continuous_line(self, polyline, color):
        line = InterpolatingLine(polyline)
        segment_num = int(line.length / PGDrivableAreaProperty.STRIPE_LENGTH)
        for segment in range(segment_num):
            start = line.get_point(PGDrivableAreaProperty.STRIPE_LENGTH * segment)

            if segment == segment_num - 1:
                end = line.get_point(line.length)
            else:
                end = line.get_point((segment + 1) * PGDrivableAreaProperty.STRIPE_LENGTH)
            node_path_list = self._construct_lane_line_segment(start, end, color, PGLineType.CONTINUOUS)
            self._node_path_list.extend(node_path_list)

    def construct_broken_line(self, polyline, color):
        line = InterpolatingLine(polyline)
        segment_num = int(line.length / (2 * PGDrivableAreaProperty.STRIPE_LENGTH))
        for segment in range(segment_num):
            start = line.get_point(segment * PGDrivableAreaProperty.STRIPE_LENGTH * 2)
            end = line.get_point(
                segment * PGDrivableAreaProperty.STRIPE_LENGTH * 2 + PGDrivableAreaProperty.STRIPE_LENGTH
            )
            if segment == segment_num - 1:
                end = line.get_point(line.length - PGDrivableAreaProperty.STRIPE_LENGTH)
            node_path_list = self._construct_lane_line_segment(start, end, color, PGLineType.BROKEN)
            self._node_path_list.extend(node_path_list)

    @property
    def block_network_type(self):
        return EdgeRoadNetwork

    def destroy(self):
        self.map_index = None
        # self.map_data = None
        super(ScenarioBlock, self).destroy()
