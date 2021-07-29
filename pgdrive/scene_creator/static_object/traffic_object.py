from typing import Tuple

import numpy as np
from panda3d.bullet import BulletCylinderShape
from panda3d.core import NodePath
from pgdrive.constants import BodyName
from pgdrive.engine.asset_loader import AssetLoader
from pgdrive.engine.physics_node import TrafficSignNode
from pgdrive.utils.coordinates_shift import panda_position, panda_heading
from pgdrive.scene_creator.static_object.base_static_object import BaseStaticObject

LaneIndex = Tuple[str, str, int]


class TrafficSign(BaseStaticObject):
    """
    Common interface for objects that appear on the road, beside vehicles.
    """
    NAME = None
    RADIUS = 0.25
    HEIGHT = 1.2
    MASS = 1

    def __init__(self, lane, lane_index: LaneIndex, longitude: float, lateral: float, random_seed):
        """
       :param lane: the lane to spawn object
        :param lane_index: the lane_index of the spawn point
        :param longitude: use to calculate cartesian position of object in the surface
        :param lateral: use to calculate the angle from positive direction of horizontal axis
        """
        position = lane.position(longitude, lateral)
        heading = lane.heading_at(longitude)
        assert self.NAME is not None, "Assign a name for this class for finding it easily"
        super(TrafficSign, self).__init__(lane, lane_index, position, heading, random_seed)
        self.position = position
        self.speed = 0
        self.heading = heading / np.pi * 180
        self.lane_index = lane_index
        self.lane = lane
        self.body_node = None

    def set_static(self, static: bool = False):
        mass = 0 if static else self.MASS
        self.body_node.setMass(mass)

    @classmethod
    def type(cls):
        return cls.__subclasses__()


class TrafficCone(TrafficSign):
    """Placed near the construction section to indicate that traffic is prohibited"""

    NAME = BodyName.Traffic_cone

    def __init__(
        self, lane, lane_index: LaneIndex, longitude: float, lateral: float, static: bool = False, random_seed=None
    ):
        super(TrafficCone, self).__init__(lane, lane_index, longitude, lateral, random_seed)
        self.body_node = TrafficSignNode(self.NAME)
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
        self.set_static(static)


class TrafficTriangle(TrafficSign):
    """Placed behind the vehicle when it breaks down"""

    NAME = BodyName.Traffic_triangle
    RADIUS = 0.5

    def __init__(
        self, lane, lane_index: LaneIndex, longitude: float, lateral: float, static: bool = False, random_seed=None
    ):
        super(TrafficTriangle, self).__init__(lane, lane_index, longitude, lateral, random_seed)
        self.body_node = TrafficSignNode(self.NAME)
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
        self.set_static(static)
