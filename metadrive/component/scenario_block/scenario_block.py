import math
from metadrive.constants import CollisionGroup
from metadrive.utils.utils import time_me
import numpy as np
from panda3d.bullet import BulletConvexHullShape, BulletTriangleMeshShape, BulletTriangleMesh
from panda3d.core import Vec3, LQuaternionf, LPoint3f

from metadrive.component.block.base_block import BaseBlock
from metadrive.component.lane.scenario_lane import ScenarioLane
from metadrive.component.road_network.edge_road_network import EdgeRoadNetwork
from metadrive.constants import DrivableAreaProperty
from metadrive.constants import PGLineType, PGLineColor
from metadrive.engine.engine_utils import get_engine
from metadrive.engine.physics_node import BulletRigidBodyNode
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.type import MetaDriveType
from metadrive.utils.coordinates_shift import panda_vector
from metadrive.utils.interpolating_line import InterpolatingLine
from metadrive.utils.math import Vector
from metadrive.utils.math import norm
from metadrive.utils.vertex import make_polygon_model


class ScenarioBlock(BaseBlock):
    LINE_CULL_DIST = 500

    def __init__(self, block_index: int, global_network, random_seed, map_index, need_lane_localization):
        # self.map_data = map_data
        self.need_lane_localization = need_lane_localization
        self.map_index = map_index
        data = self.engine.data_manager.current_scenario
        sdc_track = data.get_sdc_track()
        self.sdc_start_point = sdc_track["state"]["position"][0]
        self.crosswalks = {}
        self.sidewalks = {}
        super(ScenarioBlock, self).__init__(block_index, global_network, random_seed)

    @property
    def map_data(self):
        e = get_engine()
        return e.data_manager.get_scenario(self.map_index, should_copy=False)["map_features"]

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
            # TODO LQY: DO we need sidewalk?
            elif MetaDriveType.is_road_boundary_line(type):
                self.construct_continuous_line(np.asarray(data[ScenarioDescription.POLYLINE]), color=PGLineColor.GREY)
        self.construct_sidewalk()

    def construct_continuous_line(self, polyline, color):
        line = InterpolatingLine(polyline)
        segment_num = int(line.length / DrivableAreaProperty.STRIPE_LENGTH)
        for segment in range(segment_num):
            start = line.get_point(DrivableAreaProperty.STRIPE_LENGTH * segment)
            # trick for optimizing
            dist = norm(start[0] - self.sdc_start_point[0], start[1] - self.sdc_start_point[1])
            if dist > self.LINE_CULL_DIST:
                continue

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
            # trick for optimizing
            dist = norm(start[0] - self.sdc_start_point[0], start[1] - self.sdc_start_point[1])
            if dist > self.LINE_CULL_DIST:
                continue
            end = line.get_point(segment * DrivableAreaProperty.STRIPE_LENGTH * 2 + DrivableAreaProperty.STRIPE_LENGTH)
            if segment == segment_num - 1:
                end = line.get_point(line.length - DrivableAreaProperty.STRIPE_LENGTH)
            node_path_list = ScenarioLane.construct_lane_line_segment(self, start, end, color, PGLineType.BROKEN)
            self._node_path_list.extend(node_path_list)

    def construct_sidewalk(self):
        """
        Construct the sidewalk with collision shape
        """
        if self.engine.global_config["show_sidewalk"] and not self.engine.use_render_pipeline:
            for sidewalk in self.sidewalks.values():
                polygon = sidewalk["polygon"]
                np = make_polygon_model(polygon, 0.2)
                np.reparentTo(self.sidewalk_node_path)
                np.setPos(0, 0, 0.1)
                np.setTexture(self.ts_color, self.side_texture)
                np.setTexture(self.ts_normal, self.side_normal)

                body_node = BulletRigidBodyNode(MetaDriveType.BOUNDARY_SIDEWALK)
                body_node.setKinematic(False)
                body_node.setStatic(True)
                body_np = self.sidewalk_node_path.attachNewNode(body_node)
                body_np.setPos(0, 0, 0.1)
                self._node_path_list.append(body_np)

                geom = np.node().getGeom(0)
                mesh = BulletTriangleMesh()
                mesh.addGeom(geom)
                shape = BulletTriangleMeshShape(mesh, dynamic=False)

                body_node.addShape(shape)
                self.dynamic_nodes.append(body_node)
                body_node.setIntoCollideMask(CollisionGroup.Sidewalk)
                self._node_path_list.append(np)

    @property
    def block_network_type(self):
        return EdgeRoadNetwork

    def destroy(self):
        self.map_index = None
        # self.map_data = None
        super(ScenarioBlock, self).destroy()
