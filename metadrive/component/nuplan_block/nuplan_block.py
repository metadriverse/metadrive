import math

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
from metadrive.utils.waymo_utils.waymo_utils import convert_polyline_to_metadrive


class NuPlanBlock(BaseBlock):
    _radius = 200  # [m] show 500m map

    def __init__(self, block_index: int, global_network, random_seed, map_index):
        self.map_index = map_index
        super(NuPlanBlock, self).__init__(block_index, global_network, random_seed)

        # authorize engine access for this object
        self.engine = get_engine()
        self._nuplan_map_api = self.engine.data_manager.get_case(self.map_index).map_api
        # TODO LQY, make it a dict
        self.lines = []

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
        ]

        center = self.engine.data_manager.get_case(self.map_index).get_ego_state_at_iteration(0).center.point

        nearest_vector_map = map_api.get_proximal_map_objects(center, self._radius, layer_names)
        # Filter out stop polygons in turn stop
        if SemanticMapLayer.STOP_LINE in nearest_vector_map:
            stop_polygons = nearest_vector_map[SemanticMapLayer.STOP_LINE]
            nearest_vector_map[SemanticMapLayer.STOP_LINE] = [
                stop_polygon for stop_polygon in stop_polygons if stop_polygon.stop_line_type != StopLineType.TURN_STOP
            ]

        # Draw polygons
        polygon_layer_names = [
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.STOP_LINE,
            SemanticMapLayer.CROSSWALK,
            SemanticMapLayer.WALKWAYS,
            SemanticMapLayer.CARPARK_AREA,
        ]

        line_layer_names = [
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR
        ]
        for layer_name in polygon_layer_names:
            layer = nearest_vector_map[layer_name]
            for map_obj in layer:
                if hasattr(map_obj, "baseline_path"):
                    center_line = self._extract_centerline(map_obj)
                    # TODO (LQY) maybe using convexhull to build the collision shape in the future
                    self.block_network.add_lane(NuPlanLane(map_obj.id, center_line, width=5, lane_meta_data=map_obj))
                if layer_name in line_layer_names:
                    self.lines.append(self._extract_lane_line(map_obj.right_boundary))
                    self.lines.append(self._extract_lane_line(map_obj.left_boundary))

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
        for line in self.lines:
            self.construct_nuplan_continuous_line(
                line,
                LineColor.GREY
            )
        # draw line
        # for lane_id, data in self.nuplan_map_data.items():
        #     type = data.get("type", None)
        #     if RoadLineType.is_road_line(type):
        #         if len(data[NuPlanLaneProperty.POLYLINE]) <= 1:
        #             continue
        #         if RoadLineType.is_broken(type):
        #             self.construct_nuplan_broken_line(
        #                 convert_polyline_to_metadrive(data[NuPlanLaneProperty.POLYLINE]),
        #                 LineColor.YELLOW if RoadLineType.is_yellow(type) else LineColor.GREY
        #             )
        #         else:
        #             self.construct_nuplan_continuous_line(
        #                 convert_polyline_to_metadrive(data[NuPlanLaneProperty.POLYLINE]),
        #                 LineColor.YELLOW if RoadLineType.is_yellow(type) else LineColor.GREY
        #             )
        #     elif RoadEdgeType.is_road_edge(type) and RoadEdgeType.is_sidewalk(type):
        #         self.construct_nuplan_sidewalk(convert_polyline_to_metadrive(data[NuPlanLaneProperty.POLYLINE]))
        #     elif RoadEdgeType.is_road_edge(type) and not RoadEdgeType.is_sidewalk(type):
        #         self.construct_nuplan_continuous_line(
        #             convert_polyline_to_metadrive(data[NuPlanLaneProperty.POLYLINE]), LineColor.GREY
        #         )
        #     elif type == "center_lane" or type is None:
        #         continue
        #     # else:
        #     raise ValueError("Can not build lane line type: {}".format(type))

    def construct_nuplan_continuous_line(self, polyline, color):
        line = InterpolatingLine(polyline)
        segment_num = int(line.length / DrivableAreaProperty.STRIPE_LENGTH)
        for segment in range(segment_num):
            start = line.get_point(DrivableAreaProperty.STRIPE_LENGTH * segment)
            if segment == segment_num - 1:
                end = line.get_point(line.length)
            else:
                end = line.get_point((segment + 1) * DrivableAreaProperty.STRIPE_LENGTH)
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

    def _extract_centerline(self, map_obj):
        # TODO (LQY) Remove the offset, which is for debug
        path = map_obj.baseline_path.discrete_path
        points = [np.array([pose.x, pose.y]) for pose in path]
        return points

    def _extract_lane_line(self, boundary):
        # TODO (LQY) Remove the offset, which is for debug
        path = boundary.discrete_path
        points = [np.array([pose.x, pose.y]) for pose in path]
        return points
