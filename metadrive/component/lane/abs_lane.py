import math
from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletConvexHullShape
from panda3d.bullet import BulletGhostNode
from panda3d.core import LPoint3f
from panda3d.core import Vec3, LQuaternionf, NodePath
from panda3d.core import Vec4
from shapely import geometry

from metadrive.constants import PGDrivableAreaProperty
from metadrive.constants import MetaDriveType
from metadrive.constants import PGLineType, PGLineColor
from metadrive.engine.physics_node import BaseRigidBodyNode
from metadrive.engine.physics_node import BulletRigidBodyNode
from metadrive.utils import norm
from metadrive.utils.coordinates_shift import panda_vector, panda_heading
from metadrive.utils.math import Vector


class AbstractLane(MetaDriveType):
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

    def __init__(self, type=MetaDriveType.LANE_SURFACE_STREET):
        super(AbstractLane, self).__init__(type)
        self.speed_limit = 1000  # should be set manually
        self.index = None
        self._polygon = None
        self._shapely_polygon = None
        self.need_lane_localization = True
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
        """
        Modified from base class, the width is set to 6.5
        """
        if lane_index is not None:
            self.index = lane_index
        assert self.polygon is not None, "Polygon is required for building lane"
        # build physics contact
        if self.need_lane_localization:
            self._construct_lane_only_physics_polygon(block, self.polygon)

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
        body_node.setActive(False)
        body_node.setKinematic(False)
        body_node.setStatic(True)
        body_np = parent_np.attachNewNode(body_node)
        node_path_list.append(body_np)
        node_path_list.append(body_node)

        # its scale will change by setScale
        body_height = PGDrivableAreaProperty.LANE_LINE_GHOST_HEIGHT
        shape = BulletBoxShape(Vec3(length / 2, PGDrivableAreaProperty.LANE_LINE_WIDTH / 4, body_height))
        body_np.node().addShape(shape)
        mask = PGDrivableAreaProperty.CONTINUOUS_COLLISION_MASK if line_type != PGLineType.BROKEN else PGDrivableAreaProperty.BROKEN_COLLISION_MASK
        body_np.node().setIntoCollideMask(mask)
        block.static_nodes.append(body_np.node())

        # position and heading
        body_np.setPos(panda_vector(middle, PGDrivableAreaProperty.LANE_LINE_GHOST_HEIGHT / 2))
        direction_v = end_point - start_point
        # theta = -numpy.arctan2(direction_v[1], direction_v[0])
        theta = panda_heading(math.atan2(direction_v[1], direction_v[0]))
        body_np.setQuat(LQuaternionf(math.cos(theta / 2), 0, 0, math.sin(theta / 2)))

        return node_path_list

    def _construct_lane_only_physics_polygon(self, block, polygon):
        """
        This usually used with _construct_lane_only_vis_segment
        """
        lane = self
        # It might be Lane surface intersection
        n = BaseRigidBodyNode(lane.id, self.metadrive_type)
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

    def destroy(self):
        try:
            from metadrive.base_class.base_object import clear_node_list
        except ImportError:
            self._node_path_list.clear()
        else:
            clear_node_list(self._node_path_list)
        self._polygon = None
        self._shapely_polygon = None

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
        s_point = geometry.Point(point[0], point[1])
        return self.shapely_polygon.contains(s_point)

    @property
    def polygon(self):
        """
        Return the polygon of this lane
        Returns: a list of 2D points representing Polygon

        """
        raise NotImplementedError("Overwrite this function to allow getting polygon for this lane")

    @property
    def shapely_polygon(self):
        """Return the polygon in shapely.geometry.Polygon"""
        if self._shapely_polygon is None:
            assert self.polygon is not None
            self._shapely_polygon = geometry.Polygon(geometry.LineString(self.polygon))
        return self._shapely_polygon
