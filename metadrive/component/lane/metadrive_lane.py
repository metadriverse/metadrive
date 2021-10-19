import math

import numpy as np
from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.constants import BodyName
from metadrive.constants import DrivableAreaProperty
from metadrive.engine.physics_node import BulletRigidBodyNode
from metadrive.utils import norm
from metadrive.utils.coordinates_shift import panda_position
from metadrive.utils.math_utils import Vector
from panda3d.bullet import BulletBoxShape
from panda3d.core import Vec3, LQuaternionf


class MetaDriveLane(AbstractLane):
    radius = 0.0

    def construct_sidewalk(self, block, lateral):
        radius = self.radius
        segment_num = int(self.length / DrivableAreaProperty.SIDEWALK_LENGTH)
        for segment in range(segment_num):
            lane_start = self.position(segment * DrivableAreaProperty.SIDEWALK_LENGTH, lateral)
            if segment != segment_num - 1:
                lane_end = self.position((segment + 1) * DrivableAreaProperty.SIDEWALK_LENGTH, lateral)
            else:
                lane_end = self.position(self.length, lateral)
            self._add_sidewalk2bullet(block, lane_start, lane_end, radius, self.direction)

    @staticmethod
    def _add_sidewalk2bullet(block, lane_start, lane_end, radius=0.0, direction=0):
        middle = (lane_start + lane_end) / 2
        length = norm(lane_end[0] - lane_start[0], lane_end[1] - lane_start[1])
        body_node = BulletRigidBodyNode(BodyName.Sidewalk)
        body_node.setKinematic(False)
        body_node.setStatic(True)
        side_np = block.sidewalk_node_path.attachNewNode(body_node)
        shape = BulletBoxShape(Vec3(1 / 2, 1 / 2, 1 / 2))
        body_node.addShape(shape)
        body_node.setIntoCollideMask(block.SIDEWALK_COLLISION_MASK)
        block.dynamic_nodes.append(body_node)

        if radius == 0:
            factor = 1
        else:
            if direction == 1:
                factor = (1 - block.SIDEWALK_LINE_DIST / radius)
            else:
                factor = (1 + block.SIDEWALK_WIDTH / radius) * (1 + block.SIDEWALK_LINE_DIST / radius)
        direction_v = lane_end - lane_start
        vertical_v = Vector((-direction_v[1], direction_v[0])) / norm(*direction_v)
        middle += vertical_v * (block.SIDEWALK_WIDTH / 2 + block.SIDEWALK_LINE_DIST)
        side_np.setPos(panda_position(middle, 0))
        theta = -math.atan2(direction_v[1], direction_v[0])
        side_np.setQuat(LQuaternionf(math.cos(theta / 2), 0, 0, math.sin(theta / 2)))
        side_np.setScale(length * factor, block.SIDEWALK_WIDTH, block.SIDEWALK_THICKNESS * (1 + 0.1 * np.random.rand()))
        if block.render:
            side_np.setTexture(block.ts_color, block.side_texture)
            block.sidewalk.instanceTo(side_np)
