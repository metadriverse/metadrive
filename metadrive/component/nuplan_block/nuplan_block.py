import math
from nuplan.common.maps.nuplan_map.lane_connector import NuPlanLaneConnector

import numpy as np
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
from nuplan.planning.nuboard.style import simulation_map_layer_color

from metadrive.component.block.base_block import BaseBlock
from metadrive.component.lane.nuplan_lane import NuPlanLane
from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork
from metadrive.constants import DrivableAreaProperty
from metadrive.constants import LineColor
from metadrive.constants import LineType
from metadrive.engine.engine_utils import get_engine
from metadrive.utils.interpolating_line import InterpolatingLine
from metadrive.utils.math_utils import wrap_to_pi, norm
from dataclasses import dataclass
from metadrive.utils.waymo_utils.waymo_utils import convert_polyline_to_metadrive
from metadrive.constants import LineColor, LineType


@dataclass
class LaneLineProperty:
    points: list
    color: LineColor
    type: LineType
    in_road_connector: bool


class NuPlanBlock(BaseBlock):
    _radius = 200  # [m] show 500m map

    def __init__(self, block_index: int, global_network, random_seed, map_index):
        self.map_index = map_index
        super(NuPlanBlock, self).__init__(block_index, global_network, random_seed)

        # authorize engine access for this object
        self.engine = get_engine()
        self._nuplan_map_api = self.engine.data_manager.get_case(self.map_index).map_api
        # TODO LQY, make it a dict
        self.lines = {}

    @property
    def map_api(self):
        return self._nuplan_map_api

    def _sample_topology(self) -> bool:
        """
        This function is modified from _render_map in nuplan-devkit.simulation_tile.py
        """
        map_api = self._nuplan_map_api
        layer_names = [
            SemanticMapLayer.LANE_CONNECTOR,
            SemanticMapLayer.LANE,
            SemanticMapLayer.CROSSWALK,
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.STOP_LINE,
            SemanticMapLayer.WALKWAYS,
            SemanticMapLayer.CARPARK_AREA,
            SemanticMapLayer.ROADBLOCK,
            SemanticMapLayer.ROADBLOCK_CONNECTOR,

            # unsupported yet
            # SemanticMapLayer.STOP_SIGN,
            # SemanticMapLayer.DRIVABLE_AREA,
        ]
        center = self.engine.data_manager.get_case(self.map_index).get_ego_state_at_iteration(0).center.point

        nearest_vector_map = map_api.get_proximal_map_objects(center, self._radius, layer_names)
        # Filter out stop polygons in turn stop
        if SemanticMapLayer.STOP_LINE in nearest_vector_map:
            stop_polygons = nearest_vector_map[SemanticMapLayer.STOP_LINE]
            nearest_vector_map[SemanticMapLayer.STOP_LINE] = [
                stop_polygon for stop_polygon in stop_polygons if stop_polygon.stop_line_type != StopLineType.TURN_STOP
            ]

        # Lane and lane line
        for layer in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
            for block in nearest_vector_map[layer]:
                for lane in block.interior_edges:
                    if hasattr(lane, "baseline_path"):
                        # TODO (LQY) maybe using convexhull to build the collision shape in the future
                        self.block_network.add_lane(NuPlanLane(lane_meta_data=lane))
                        is_connector = True if layer == SemanticMapLayer.ROADBLOCK_CONNECTOR else False
                        self._get_lane_line(lane, is_road_connector=is_connector)

        # walkway (Disable currently)
        # for walkway in nearest_vector_map[SemanticMapLayer.WALKWAYS]:
        #     self._get_sidewalk(walkway)
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
        for line_prop in self.lines.values():
            if line_prop.type == LineType.CONTINUOUS:
                self.construct_nuplan_continuous_line(
                    line_prop.points,
                    line_prop.color)
            if line_prop.type == LineType.BROKEN:
                self.construct_nuplan_broken_line(
                    line_prop.points,
                    line_prop.color)
            # if line_prop.type == LineType.SIDE:
            #     self.construct_nuplan_sidewalk(
            #         line_prop.points)

    def construct_nuplan_continuous_line(self, polyline, color):
        for start, end, in zip(polyline[:-1], polyline[1:]):
            node_path_list = NuPlanLane.construct_lane_line_segment(self, start, end, color, LineType.CONTINUOUS)
            self._node_path_list.extend(node_path_list)

    def construct_nuplan_broken_line(self, polyline, color):
        line = InterpolatingLine(polyline)
        segment_num = int(line.length / (2 * DrivableAreaProperty.STRIPE_LENGTH))
        for segment in range(segment_num):
            start = line.get_point(segment * DrivableAreaProperty.STRIPE_LENGTH * 2)
            end = line.get_point(segment * DrivableAreaProperty.STRIPE_LENGTH * 2 + DrivableAreaProperty.STRIPE_LENGTH)
            if segment == segment_num - 1:
                end = line.get_point(line.length - DrivableAreaProperty.STRIPE_LENGTH)
            node_path_list = NuPlanLane.construct_lane_line_segment(self, start, end, color, LineType.BROKEN)
            self._node_path_list.extend(node_path_list)

    def construct_nuplan_sidewalk(self, polyline):
        line = InterpolatingLine(polyline)
        seg_len = DrivableAreaProperty.LANE_SEGMENT_LENGTH
        segment_num = int(line.length / seg_len)
        last_theta = None
        for segment in range(segment_num):
            lane_start = line.get_point(segment * seg_len)
            lane_end = line.get_point((segment + 1) * seg_len)
            if segment == segment_num - 1:
                lane_end = line.get_point(line.length)
            direction_v = lane_end - lane_start
            theta = -math.atan2(direction_v[1], direction_v[0])
            if segment == segment_num - 1:
                factor = 1
            else:
                factor = 1.25
                if last_theta is not None:
                    diff = wrap_to_pi(theta) - wrap_to_pi(last_theta)
                    if diff > 0:
                        factor += np.sin(abs(diff) / 2) * DrivableAreaProperty.SIDEWALK_WIDTH / norm(
                            lane_start[0] - lane_end[0], lane_start[1] - lane_end[1]
                        ) + 0.15
                    else:
                        factor -= np.sin(abs(diff) / 2) * DrivableAreaProperty.SIDEWALK_WIDTH / norm(
                            lane_start[0] - lane_end[0], lane_start[1] - lane_end[1]
                        )
            last_theta = theta
            node_path_list = NuPlanLane.construct_sidewalk_segment(
                self, lane_start, lane_end, length_multiply=factor, extra_thrust=1
            )
            self._node_path_list.extend(node_path_list)

    @property
    def block_network_type(self):
        return EdgeRoadNetwork

    def destroy(self):
        self.map_index = None
        self.nuplan_map_data = None
        super(NuPlanBlock, self).destroy()

    def __del__(self):
        self.destroy()
        super(NuPlanBlock, self).__del__()
        print("NuPlan Block is being deleted.")

    # @property
    # def nuplan_map_data(self):
    #     e = get_engine()
    #     return e.data_manager.get_case(self.map_index)["map"]

    def _get_lane_line(self, lane, is_road_connector=False):
        boundaries = self.map_api._get_vector_map_layer(13)

        def _get_points(boundary):
            path = boundary.discrete_path
            points = [np.array([pose.x, pose.y]) for pose in path]
            return points

        if not is_road_connector:
            right = lane.right_boundary
            if right.id not in self.lines:
                line_type = self._metadrive_line_type(int(boundaries.loc[[str(right.id)]]["boundary_type_fid"]))
                self.lines[right.id] = LaneLineProperty(_get_points(right), LineColor.GREY, line_type,
                                                        in_road_connector=is_road_connector)

            left = lane.left_boundary
            if left.id not in self.lines:
                line_type = self._metadrive_line_type(int(boundaries.loc[[str(left.id)]]["boundary_type_fid"]))
                line_color = LineColor.YELLOW if lane.adjacent_edges[0] is None and lane.adjacent_edges[
                    1] is not None else LineColor.GREY
                self.lines[left.id] = LaneLineProperty(_get_points(left), line_color, line_type,
                                                       in_road_connector=is_road_connector)

        # else:
        # (Disable currently)
        #     right: NuPlanLaneConnector = lane.right_boundary
        #     left: NuPlanLaneConnector = lane.left_boundary
        #     if left.id not in self.lines:
        #         if lane.incoming_edges[0].adjacent_edges[0] is None or lane.outgoing_edges[0].adjacent_edges[0] is None:
        #             self.lines[left.id] = LaneLineProperty(_get_points(left), LineColor.YELLOW, LineType.CONTINUOUS,
        #                                                    in_road_connector=is_road_connector)
        #     if right.id not in self.lines:
        #         if lane.incoming_edges[-1].adjacent_edges[-1] is None \
        #                 or lane.outgoing_edges[-1].adjacent_edges[-1] is None:
        #             self.lines[right.id] = LaneLineProperty(_get_points(right), LineColor.GREY, LineType.CONTINUOUS,
        #                                                     in_road_connector=is_road_connector)

    def _get_sidewalk(self, walkway):
        if walkway.id not in self.lines:
            points = np.array(
                [i for i in zip(walkway.polygon.boundary.coords.xy[0], walkway.polygon.boundary.coords.xy[1])])
            self.lines[walkway.id] = LaneLineProperty(points, LineColor.GREY, LineType.SIDE,
                                                      in_road_connector=False)

    def _get_lane_boundary(self, line):
        pass

    def _metadrive_line_type(self, nuplan_type):
        if nuplan_type == 2:
            return LineType.CONTINUOUS
        elif nuplan_type == 0:
            return LineType.BROKEN
        elif nuplan_type == 3:
            return LineType.NONE
        else:
            raise ValueError("Unknown line tyep: {}".format(nuplan_type))
