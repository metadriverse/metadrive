import math
from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletConvexHullShape
from panda3d.bullet import BulletGhostNode
from panda3d.core import LPoint3f
from panda3d.core import Vec3, LQuaternionf, CardMaker, NodePath
from panda3d.core import Vec4

from metadrive.constants import DrivableAreaProperty
from metadrive.constants import MetaDriveType
from metadrive.constants import PGLineType, PGLineColor
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.physics_node import BaseRigidBodyNode
from metadrive.engine.physics_node import BulletRigidBodyNode
from metadrive.utils import norm
from metadrive.utils.coordinates_shift import panda_vector, panda_heading
from metadrive.utils.math import Vector


class AbstractLane:
    """A lane on the road, described by its central curve."""

    metaclass__ = ABCMeta
    DEFAULT_WIDTH: float = 4
    line_types: Tuple[PGLineType, PGLineType]
    line_colors = [PGLineColor.GREY, PGLineColor.GREY]
    length = 0
    start = None
    end = None
    VEHICLE_LENGTH = 4
    _RANDOM_HEIGHT_OFFSET = np.arange(0, 0.02, 0.0005)
    _RANDOM_HEIGHT_OFFSET_INDEX = 0

    def __init__(self):
        self.speed_limit = 1000  # should be set manually
        self.index = None

        self._node_path_list = []

    def set_speed_limit(self, speed_limit):
        self.speed_limit = speed_limit

    @abstractmethod
    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        """
        Convert local lane coordinates to a physx_world position.

        :param longitudinal: longitudinal lane coordinate [m]
        :param lateral: lateral lane coordinate [m]
        :return: the corresponding physx_world position [m]
        """
        raise NotImplementedError()

    @abstractmethod
    def local_coordinates(self, position: np.ndarray) -> Tuple[float, float]:
        """
        Convert a physx_world position to local lane coordinates.

        :param position: a physx_world position [m]
        :return: the (longitudinal, lateral) lane coordinates [m]
        """
        raise NotImplementedError()

    @abstractmethod
    def heading_theta_at(self, longitudinal: float) -> float:
        """
        Get the lane heading at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane heading [rad]
        """
        raise NotImplementedError()

    def heading_at(self, longitudinal) -> np.array:
        heaidng_theta = self.heading_theta_at(longitudinal)
        return np.array([math.cos(heaidng_theta), math.sin(heaidng_theta)])

    @abstractmethod
    def width_at(self, longitudinal: float) -> float:
        """
        Get the lane width at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane width [m]
        """
        raise NotImplementedError()

    def on_lane(self, position: np.ndarray, longitudinal: float = None, lateral: float = None, margin: float = 0) \
            -> bool:
        """
        Whether a given physx_world position is on the lane.

        :param position: a physx_world position [m]
        :param longitudinal: (optional) the corresponding longitudinal lane coordinate, if known [m]
        :param lateral: (optional) the corresponding lateral lane coordinate, if known [m]
        :param margin: (optional) a supplementary margin around the lane width
        :return: is the position on the lane?
        """
        if not longitudinal or not lateral:
            longitudinal, lateral = self.local_coordinates(position)
        is_on = math.fabs(lateral) <= self.width_at(longitudinal) / 2 + margin and \
                -self.VEHICLE_LENGTH <= longitudinal < self.length + self.VEHICLE_LENGTH
        return is_on

    def distance(self, position):
        """Compute the L1 distance [m] from a position to the lane."""
        s, r = self.local_coordinates(position)
        a = s - self.length
        b = 0 - s
        # return abs(r) + max(s - self.length, 0) + max(0 - s, 0)
        return abs(r) + (a if a > 0 else 0) + (b if b > 0 else 0)

    def is_previous_lane_of(self, target_lane, error_region=1e-1):
        x_1, y_1 = self.end
        x_2, y_2 = target_lane.start
        if norm(x_1 - x_2, y_1 - y_2) < error_region:
            return True
        return False

    def construct_lane_in_block(self, block, lane_index):
        segment_num = int(self.length / DrivableAreaProperty.LANE_SEGMENT_LENGTH)
        if segment_num == 0:
            middle = self.position(self.length / 2, 0)
            end = self.position(self.length, 0)
            theta = self.heading_theta_at(self.length / 2)
            width = self.width_at(0) + DrivableAreaProperty.SIDEWALK_LINE_DIST * 2
            self.construct_lane_segment(block, middle, width, self.length, theta, lane_index)
        for i in range(segment_num):
            middle = self.position(self.length * (i + .5) / segment_num, 0)
            end = self.position(self.length * (i + 1) / segment_num, 0)
            direction_v = end - middle
            theta = math.atan2(direction_v[1], direction_v[0])
            width = self.width_at(0) + DrivableAreaProperty.SIDEWALK_LINE_DIST * 2
            length = self.length
            self.construct_lane_segment(block, middle, width, length * 1.3 / segment_num, theta, lane_index)

    def construct_lane_line_in_block(self, block, construct_left_right=(True, True)):
        """
        Construct lane line in the Panda3d world for getting contact information
        """
        for idx, line_type, line_color, need, in zip([-1, 1], self.line_types, self.line_colors, construct_left_right):
            if not need:
                continue
            lateral = idx * self.width_at(0) / 2
            if line_type == PGLineType.CONTINUOUS:
                self.construct_continuous_line(block, lateral, line_color, line_type)
            elif line_type == PGLineType.BROKEN:
                self.construct_broken_line(block, lateral, line_color, line_type)
            elif line_type == PGLineType.SIDE:
                self.construct_continuous_line(block, lateral, line_color, line_type)
                self.construct_sidewalk(block, lateral)
            elif line_type == PGLineType.NONE:
                continue
            else:
                raise ValueError(
                    "You have to modify this cuntion and implement a constructing method for line type: {}".
                    format(line_type)
                )

    def construct_broken_line(self, block, lateral, line_color, line_type):
        """
        Lateral: left[-1/2 * width] or right[1/2 * width]
        """
        segment_num = int(self.length / (2 * DrivableAreaProperty.STRIPE_LENGTH))
        for segment in range(segment_num):
            start = self.position(segment * DrivableAreaProperty.STRIPE_LENGTH * 2, lateral)
            end = self.position(
                segment * DrivableAreaProperty.STRIPE_LENGTH * 2 + DrivableAreaProperty.STRIPE_LENGTH, lateral
            )
            if segment == segment_num - 1:
                end = self.position(self.length - DrivableAreaProperty.STRIPE_LENGTH, lateral)
            node_path_list = self.construct_lane_line_segment(block, start, end, line_color, line_type)
            self._node_path_list.extend(node_path_list)

    def construct_continuous_line(self, block, lateral, line_color, line_type):
        """
        We process straight line to several pieces by default, which can be optimized through overriding this function
        Lateral: left[-1/2 * width] or right[1/2 * width]
        """
        segment_num = int(self.length / DrivableAreaProperty.LANE_SEGMENT_LENGTH)
        if segment_num == 0:
            start = self.position(0, lateral)
            end = self.position(self.length, lateral)
            node_path_list = self.construct_lane_line_segment(block, start, end, line_color, line_type)
            self._node_path_list.extend(node_path_list)
        for segment in range(segment_num):
            start = self.position(DrivableAreaProperty.LANE_SEGMENT_LENGTH * segment, lateral)
            if segment == segment_num - 1:
                end = self.position(self.length, lateral)
            else:
                end = self.position((segment + 1) * DrivableAreaProperty.LANE_SEGMENT_LENGTH, lateral)
            node_path_list = self.construct_lane_line_segment(block, start, end, line_color, line_type)
            self._node_path_list.extend(node_path_list)

    def construct_sidewalk(self, block, lateral):
        """
        Lateral: left[-1/2 * width] or right[1/2 * width]
        """
        pass

    def construct_lane_segment(self, block, position, width, length, theta, lane_index):
        """
        Construct a PART of this lane in block. The reason for using this is that we can use box shape to apporximate
        almost all shapes

        :param block: it should be constructed in block
        :param position: Middle point
        :param width: Lane width
        :param length: Segment length
        :param theta: Rotate theta
        :param lane_index: set index for this lane, sometimes lane index is decided after building graph
        """
        theta = panda_heading(theta)
        lane = self
        length += 0.1
        if lane_index is not None:
            lane.index = lane_index

        n = BaseRigidBodyNode(lane.index, MetaDriveType.LANE_SURFACE_STREET)
        segment_np = NodePath(n)

        self._node_path_list.append(segment_np)
        self._node_path_list.append(n)

        segment_node = segment_np.node()
        segment_node.set_active(False)
        segment_node.setKinematic(False)
        segment_node.setStatic(True)
        shape = BulletBoxShape(Vec3(length / 2, 0.1, width / 2))
        segment_node.addShape(shape)
        block.static_nodes.append(segment_node)
        segment_np.setPos(panda_vector(position, -0.1))

        segment_np.setH(theta / np.pi * 180)
        segment_np.setP(-90)
        segment_np.reparentTo(block.lane_node_path)
        if block.render and not block.use_render_pipeline:
            cm = CardMaker('card')
            cm.setFrame(-length / 2, length / 2, -width / 2, width / 2)
            cm.setHasNormals(True)
            cm.setUvRange((-length / 5, -width / 5), (length / 5, width / 5))
            card = block.lane_vis_node_path.attachNewNode(cm.generate())
            self._node_path_list.append(card)

            card.setPos(panda_vector(position, np.random.rand() * 0.02 - 0.015))

            card.setH(theta / np.pi * 180)
            card.setP(-90)
            # card.setTransparency(TransparencyAttrib.MMultisample)
            # card.setTexture(block.ts_normal, block.road_normal)
            card.setTexture(block.ts_color, block.road_texture)

    @staticmethod
    def construct_lane_line_segment(block, start_point, end_point, line_color: Vec4, line_type: PGLineType):
        node_path_list = []
        # static_node_list = []
        # dynamic_node_list = []

        if not isinstance(start_point, np.ndarray):
            start_point = np.array(start_point)
        if not isinstance(end_point, np.ndarray):
            end_point = np.array(end_point)

        length = norm(end_point[0] - start_point[0], end_point[1] - start_point[1])
        middle = (start_point + end_point) / 2
        parent_np = block.lane_line_node_path
        if length <= 0:
            return []
        if PGLineType.prohibit(line_type):
            node_name = MetaDriveType.LINE_SOLID_SINGLE_WHITE if line_color == PGLineColor.GREY else MetaDriveType.LINE_SOLID_SINGLE_YELLOW
        else:
            # node_name = MetaDriveType.LINE_SOLID_SINGLE_WHITE if line_color == PGLineColor.GREY else MetaDriveType.LINE_SOLID_SINGLE_YELLOW
            node_name = MetaDriveType.LINE_BROKEN_SINGLE_WHITE if line_color == PGLineColor.GREY else MetaDriveType.LINE_BROKEN_SINGLE_YELLOW

        # add bullet body for it
        body_node = BulletGhostNode(node_name)
        body_node.set_active(False)
        body_node.setKinematic(False)
        body_node.setStatic(True)
        body_np = parent_np.attachNewNode(body_node)
        node_path_list.append(body_np)
        node_path_list.append(body_node)

        # its scale will change by setScale
        body_height = DrivableAreaProperty.LANE_LINE_GHOST_HEIGHT
        shape = BulletBoxShape(Vec3(length / 2, DrivableAreaProperty.LANE_LINE_WIDTH / 4, body_height))
        body_np.node().addShape(shape)
        mask = DrivableAreaProperty.CONTINUOUS_COLLISION_MASK if line_type != PGLineType.BROKEN else DrivableAreaProperty.BROKEN_COLLISION_MASK
        body_np.node().setIntoCollideMask(mask)
        block.static_nodes.append(body_np.node())

        # position and heading
        body_np.setPos(panda_vector(middle, DrivableAreaProperty.LANE_LINE_GHOST_HEIGHT / 2))
        direction_v = end_point - start_point
        # theta = -numpy.arctan2(direction_v[1], direction_v[0])
        theta = panda_heading(math.atan2(direction_v[1], direction_v[0]))
        body_np.setQuat(LQuaternionf(math.cos(theta / 2), 0, 0, math.sin(theta / 2)))

        if block.render and not block.use_render_pipeline:
            # For visualization
            lane_line = block.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
            lane_line.setTwoSided(False)
            lane_line.setScale(length, DrivableAreaProperty.LANE_LINE_WIDTH, DrivableAreaProperty.LANE_LINE_THICKNESS)
            height = 0
            lane_line.setTexture(block.ts_color, block.lane_line_texture)
            height += 0.01 if line_color == PGLineColor.YELLOW else 0
            lane_line.setQuat(LQuaternionf(math.cos(theta / 2), 0, 0, math.sin(theta / 2)))
            lane_line.setPos(panda_vector(middle, height))
            lane_line.reparentTo(parent_np)
            lane_line.set_color(line_color)

        return node_path_list

    @staticmethod
    def construct_sidewalk_segment(block, lane_start, lane_end, length_multiply=1, extra_thrust=0, width=0):
        node_path_list = []

        direction_v = lane_end - lane_start
        if abs(norm(direction_v[0], direction_v[1])) < 0.1:
            return []
        width = width or block.SIDEWALK_WIDTH
        middle = (lane_start + lane_end) / 2
        length = norm(lane_end[0] - lane_start[0], lane_end[1] - lane_start[1])
        body_node = BulletRigidBodyNode(MetaDriveType.BOUNDARY_LINE)
        body_node.setKinematic(False)
        body_node.setStatic(True)
        side_np = block.sidewalk_node_path.attachNewNode(body_node)
        node_path_list.append(side_np)

        shape = BulletBoxShape(Vec3(1 / 2, 1 / 2, 1 / 2))
        body_node.addShape(shape)
        body_node.setIntoCollideMask(block.SIDEWALK_COLLISION_MASK)
        if block.render:
            # a trick to acc off-rendering training
            block.dynamic_nodes.append(body_node)
        else:
            block.static_nodes.append(body_node)

        direction_v = lane_end - lane_start
        if extra_thrust != 0:
            vertical_v = Vector((direction_v[1], -direction_v[0])) / norm(*direction_v)
            middle += vertical_v * extra_thrust
        side_np.setPos(panda_vector(middle, 0))
        theta = math.atan2(direction_v[1], direction_v[0])
        side_np.setQuat(LQuaternionf(math.cos(theta / 2), 0, 0, math.sin(theta / 2)))
        side_np.setScale(length * length_multiply, width, block.SIDEWALK_THICKNESS * (1 + 0.1 * np.random.rand()))
        if block.render and not block.use_render_pipeline:
            block.sidewalk.instanceTo(side_np)

        return node_path_list

    def _construct_lane_only_physics_polygon(self, block, polygon):
        """
        This usually used with _construct_lane_only_vis_segment
        """
        lane = self
        n = BaseRigidBodyNode(lane.id, MetaDriveType.LANE_SURFACE_STREET)
        segment_np = NodePath(n)

        self._node_path_list.append(segment_np)
        self._node_path_list.append(n)

        segment_node = segment_np.node()
        segment_node.set_active(False)
        segment_node.setKinematic(False)
        segment_node.setStatic(True)
        shape = BulletConvexHullShape()
        for point in polygon:
            # Panda coordinate is different from metadrive coordinate
            point_up = LPoint3f(*point, 0.0)
            shape.addPoint(LPoint3f(*point_up))
            point_down = LPoint3f(*point, -0.1)
            shape.addPoint(LPoint3f(*point_down))
        segment_node.addShape(shape)
        block.static_nodes.append(segment_node)
        segment_np.reparentTo(block.lane_node_path)

    def _construct_lane_only_vis_segment(self, block, position, width, length, theta):
        """
        Only create visual part for this lane, usually used with _construct_lane_only_physics_polygon()
        """
        # The lane surface is created with terrain now
        if block.render and not block.use_render_pipeline:
            length += 0.1
            theta = panda_heading(theta)
            cm = CardMaker('card')
            cm.setFrame(-length / 2, length / 2, -width / 2, width / 2)
            cm.setHasNormals(True)
            cm.setUvRange((0, 0), (length / 20, width / 10))
            card = block.lane_vis_node_path.attachNewNode(cm.generate())
            self._node_path_list.append(card)

            card.setPos(panda_vector(position, np.random.rand() * 0.02 - 0.015))

            card.setH(theta / np.pi * 180)
            card.setP(-90)
            # card.setTransparency(TransparencyAttrib.MMultisample)
            card.setTexture(block.ts_color, block.road_texture)

    def destroy(self):
        try:
            from metadrive.base_class.base_object import clear_node_list
        except ImportError:
            self._node_path_list.clear()
        else:
            clear_node_list(self._node_path_list)

    def __del__(self):
        self.destroy()

    def get_polyline(self, interval=2, lateral=0):
        """
        This method will return the center line of this Lane in a discrete vector representation
        """
        ret = []
        for i in np.arange(0, self.length, interval):
            ret.append(self.position(i, lateral))
        ret.append(self.position(self.length, lateral))
        return np.array(ret)

    @property
    def id(self):
        return self.index

    def point_on_lane(self, point):
        """
        Return True if the point is in the lane polygon
        """
        raise NotImplementedError

    @property
    def polygon(self):
        raise NotImplementedError
