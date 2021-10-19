import math
from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np
from metadrive.constants import BodyName
from metadrive.constants import DrivableAreaProperty
from metadrive.constants import LineType, LineColor
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.physics_node import BaseRigidBodyNode
from metadrive.utils import norm
from metadrive.utils.coordinates_shift import panda_position
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletGhostNode
from panda3d.core import Vec3, LQuaternionf, CardMaker, TransparencyAttrib, NodePath
from panda3d.core import Vec4


class AbstractLane:
    """A lane on the road, described by its central curve."""

    metaclass__ = ABCMeta
    DEFAULT_WIDTH: float = 4
    line_types: Tuple[LineType, LineType]
    line_colors = [LineColor.GREY, LineColor.GREY]

    def __init__(self):
        self.speed_limit = 1000  # should be set manually
        self.index = None

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

    def heading_at(self, longitudinal) -> list:
        heaidng_theta = self.heading_theta_at(longitudinal)
        return [math.cos(heaidng_theta), math.sin(heaidng_theta)]

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

    def is_previous_lane_of(self, target_lane):
        x_1, y_1 = self.end
        x_2, y_2 = target_lane.start
        if norm(x_1 - x_2, y_1 - y_2) < 1e-1:
            return True
        return False

    def construct_lane_in_block(self, block, lane_index=None):
        """
        Every lane should implement this method for constructing it in the panda3D world
        """
        raise NotImplementedError

    def construct_lane_line_in_block(self, block):
        """
        Construct lane line in the Panda3d world for getting contact information
        """
        for idx, line_type, line_color in zip([-1, 1], self.line_types, self.line_colors):
            lateral = idx * self.width_at(0) / 2
            if line_type == LineType.CONTINUOUS:
                self.construct_continuous_line(block,lateral, line_color, line_type)
            elif line_type == LineType.BROKEN:
                self.construct_broken_line(block, lateral, line_color, line_type)
            elif line_type == LineType.SIDE:
                # self.construct_sidewalk(block, lateral, line_color, line_type)
                self.construct_continuous_line(block, lateral, line_color, line_type)
            else:
                raise ValueError(
                    "You have to modify this cuntion and implement a constructing method for line type: {}".format(
                        line_type))

    def construct_continuous_line(self, block, lateral, line_color, line_type):
        """
        Lateral: left[-1/2 * width] or right[1/2 * width]
        """
        segment_num = int(self.length / (2 * DrivableAreaProperty.STRIPE_LENGTH))
        for segment in range(segment_num):
            start = self.position(segment * DrivableAreaProperty.STRIPE_LENGTH * 2, lateral)
            end = self.position(
                segment * DrivableAreaProperty.STRIPE_LENGTH * 2 + DrivableAreaProperty.STRIPE_LENGTH,
                lateral
            )
            self.construct_lane_line_segment(block, start, end, line_color, line_type)

        start = self.position(segment_num * DrivableAreaProperty.STRIPE_LENGTH * 2, lateral)
        end = self.position(self.length + DrivableAreaProperty.STRIPE_LENGTH, lateral)
        self.construct_lane_line_segment(block, start, end, line_color, line_type)


    def construct_broken_line(self, block, lateral, line_color, line_type):
        """
        Lateral: left[-1/2 * width] or right[1/2 * width]
        """
        segment_num = int(self.length / (2 * DrivableAreaProperty.STRIPE_LENGTH))
        for segment in range(segment_num):
            start = self.position(segment * DrivableAreaProperty.STRIPE_LENGTH * 2, lateral)
            end = self.position(
                segment * DrivableAreaProperty.STRIPE_LENGTH * 2 + DrivableAreaProperty.STRIPE_LENGTH,
                lateral
            )
            self.construct_lane_line_segment(block, start, end, line_color, line_type)

        start = self.position(segment_num * DrivableAreaProperty.STRIPE_LENGTH * 2, lateral)
        end = self.position(self.length + DrivableAreaProperty.STRIPE_LENGTH, lateral)
        self.construct_lane_line_segment(block, start, end, line_color, line_type)

    def construct_sidewalk(self, block, lateral, line_color, line_type):
        """
        Lateral: left[-1/2 * width] or right[1/2 * width]
        """
        segment_num = int(self.length / (2 * DrivableAreaProperty.STRIPE_LENGTH))
        for segment in range(segment_num):
            start = self.position(segment * DrivableAreaProperty.STRIPE_LENGTH * 2, lateral)
            end = self.position(
                segment * DrivableAreaProperty.STRIPE_LENGTH * 2 + DrivableAreaProperty.STRIPE_LENGTH,
                lateral
            )
            self.construct_lane_line_segment(block, start, end, line_color, line_type)

        start = self.position(segment_num * DrivableAreaProperty.STRIPE_LENGTH * 2, lateral)
        end = self.position(self.length + DrivableAreaProperty.STRIPE_LENGTH, lateral)
        self.construct_lane_line_segment(block, start, end, line_color, line_type)

    def construct_lane_segment(self, block, position, width, length, theta, lane_index=None):
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
        lane = self
        length += 0.1
        if lane_index is not None:
            lane.index = lane_index
        segment_np = NodePath(BaseRigidBodyNode(lane, BodyName.Lane))
        segment_node = segment_np.node()
        segment_node.set_active(False)
        segment_node.setKinematic(False)
        segment_node.setStatic(True)
        shape = BulletBoxShape(Vec3(length / 2, 0.1, width / 2))
        segment_node.addShape(shape)
        block.static_nodes.append(segment_node)
        segment_np.setPos(panda_position(position, -0.1))
        segment_np.setQuat(
            LQuaternionf(
                math.cos(theta / 2) * math.cos(-math.pi / 4),
                math.cos(theta / 2) * math.sin(-math.pi / 4), -math.sin(theta / 2) * math.cos(-math.pi / 4),
                math.sin(theta / 2) * math.cos(-math.pi / 4)
            )
        )
        segment_np.reparentTo(block.lane_node_path)
        if block.render:
            cm = CardMaker('card')
            cm.setFrame(-length / 2, length / 2, -width / 2, width / 2)
            cm.setHasNormals(True)
            cm.setUvRange((0, 0), (length / 20, width / 10))
            card = block.lane_vis_node_path.attachNewNode(cm.generate())
            card.setPos(panda_position(position, np.random.rand() * 0.01 - 0.01))

            card.setQuat(
                LQuaternionf(
                    math.cos(theta / 2) * math.cos(-math.pi / 4),
                    math.cos(theta / 2) * math.sin(-math.pi / 4), -math.sin(theta / 2) * math.cos(-math.pi / 4),
                    math.sin(theta / 2) * math.cos(-math.pi / 4)
                )
            )
            card.setTransparency(TransparencyAttrib.MMultisample)
            card.setTexture(block.ts_color, block.road_texture)

    @staticmethod
    def construct_lane_line_segment(
            block,
            start_point,
            end_point,
            color: Vec4,
            line_type: LineType,
    ):
        length = norm(end_point[0] - start_point[0], end_point[1] - start_point[1])
        middle = (start_point + end_point) / 2
        parent_np = block.lane_line_node_path
        if length <= 0:
            return
        if LineType.prohibit(line_type):
            node_name = BodyName.White_continuous_line if color == LineColor.GREY else BodyName.Yellow_continuous_line
        else:
            node_name = BodyName.Broken_line

        # add bullet body for it
        body_node = BulletGhostNode(node_name)
        body_node.set_active(False)
        body_node.setKinematic(False)
        body_node.setStatic(True)
        body_np = parent_np.attachNewNode(body_node)
        # its scale will change by setScale
        body_height = DrivableAreaProperty.LANE_LINE_GHOST_HEIGHT
        shape = BulletBoxShape(
            Vec3(
                length / 2 if line_type != LineType.BROKEN else length, DrivableAreaProperty.LANE_LINE_WIDTH / 2,
                body_height
            )
        )
        body_np.node().addShape(shape)
        mask = DrivableAreaProperty.CONTINUOUS_COLLISION_MASK if line_type != LineType.BROKEN else DrivableAreaProperty.BROKEN_COLLISION_MASK
        body_np.node().setIntoCollideMask(mask)
        block.static_nodes.append(body_np.node())

        # position and heading
        body_np.setPos(panda_position(middle, DrivableAreaProperty.LANE_LINE_GHOST_HEIGHT / 2))
        direction_v = end_point - start_point
        # theta = -numpy.arctan2(direction_v[1], direction_v[0])
        theta = -math.atan2(direction_v[1], direction_v[0])
        body_np.setQuat(LQuaternionf(math.cos(theta / 2), 0, 0, math.sin(theta / 2)))

        if block.render:
            # For visualization
            lane_line = block.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
            lane_line.setScale(length, DrivableAreaProperty.LANE_LINE_WIDTH, DrivableAreaProperty.LANE_LINE_THICKNESS)
            lane_line.setPos(Vec3(0, 0 - DrivableAreaProperty.LANE_LINE_GHOST_HEIGHT / 2))
            lane_line.reparentTo(body_np)
            body_np.set_color(color)
