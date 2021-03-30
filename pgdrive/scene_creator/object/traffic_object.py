from typing import Sequence, Tuple

import numpy as np
from panda3d.bullet import BulletRigidBodyNode, BulletCylinderShape
from panda3d.core import NodePath

from pgdrive.constants import BodyName
from pgdrive.utils.asset_loader import AssetLoader
from pgdrive.utils.coordinates_shift import panda_position, panda_heading
from pgdrive.utils.element import Element

LaneIndex = Tuple[str, str, int]


class ObjectNode(BulletRigidBodyNode):
    """
    Collision Properties should place here, info here can used for collision callback
    """
    COST_ONCE = True  # cost will give at the first time

    def __init__(self, object_body_name: str):
        BulletRigidBodyNode.__init__(self, object_body_name)
        BulletRigidBodyNode.setPythonTag(self, object_body_name, self)
        self.crashed = False


class Object(Element):
    """
    Common interface for objects that appear on the road, beside vehicles.
    """
    NAME = None
    RADIUS = 0.25
    HEIGHT = 1.2
    MASS = 1

    def __init__(self, lane, lane_index: LaneIndex, position: Sequence[float], heading: float = 0.):
        """
       :param lane: the lane to spawn object
        :param lane_index: the lane_index of the spawn point
        :param position: cartesian position of object in the surface
        :param heading: the angle from positive direction of horizontal axis
        """
        assert self.NAME is not None, "Assign a name for this class for finding it easily"
        super(Object, self).__init__()
        self.position = position
        self.speed = 0
        self.heading = heading / np.pi * 180
        self.lane_index = lane_index
        self.lane = lane
        self.body_node = None

    @classmethod
    def make_on_lane(cls, lane, lane_index: LaneIndex, longitudinal: float, lateral: float):
        """
        Create an object on a given lane at a longitudinal position.

        :param lane: the lane to spawn object
        :param lane_index: the lane_index of the spawn point
        :param longitudinal: longitudinal position along the lane
        :param lateral: lateral position
        :return: An object with at the specified position
        """
        return cls(lane, lane_index, lane.position(longitudinal, lateral), lane.heading_at(longitudinal))

    def set_static(self, static: bool = False):
        mass = 0 if static else self.MASS
        self.body_node.setMass(mass)

    @classmethod
    def type(cls):
        return cls.__subclasses__()


class TrafficCone(Object):
    """Placed near the construction section to indicate that traffic is prohibited"""

    NAME = BodyName.Traffic_cone

    def __init__(self, lane, lane_index: LaneIndex, position: Sequence[float], heading: float = 0.):
        super(TrafficCone, self).__init__(lane, lane_index, position, heading)
        self.body_node = ObjectNode(self.NAME)
        self.body_node.addShape(BulletCylinderShape(self.RADIUS, self.HEIGHT))
        self.node_path: NodePath = NodePath(self.body_node)
        self.node_path.setPos(panda_position(self.position, self.HEIGHT / 2))
        self.dynamic_nodes.append(self.body_node)
        self.node_path.setH(panda_heading(self.heading))
        if self.render:
            model = self.loader.loadModel(AssetLoader.file_path("models", "traffic_cone", "scene.gltf"))
            model.setScale(0.02)
            model.setPos(0, 0, -self.HEIGHT / 2)
            model.reparentTo(self.node_path)


class TrafficTriangle(Object):
    """Placed behind the vehicle when it breaks down"""

    NAME = BodyName.Traffic_triangle

    def __init__(self, lane, lane_index: LaneIndex, position: Sequence[float], heading: float = 0.):
        super(TrafficTriangle, self).__init__(lane, lane_index, position, heading)
        self.body_node = ObjectNode(self.NAME)
        self.body_node.addShape(BulletCylinderShape(self.RADIUS, self.HEIGHT))
        self.node_path: NodePath = NodePath(self.body_node)
        self.node_path.setPos(panda_position(self.position, self.HEIGHT / 2))
        self.dynamic_nodes.append(self.body_node)
        self.node_path.setH(panda_heading(self.heading))
        if self.render:
            model = self.loader.loadModel(AssetLoader.file_path("models", "warning", "warning.gltf"))
            model.setScale(0.02)
            model.setH(-90)
            model.setPos(0, 0, -self.HEIGHT / 2)
            model.reparentTo(self.node_path)
