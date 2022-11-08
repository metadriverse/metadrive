from typing import Tuple

from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletCylinderShape

from metadrive.component.static_object.base_static_object import BaseStaticObject
from metadrive.constants import BodyName
from metadrive.constants import CollisionGroup
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.physics_node import BaseRigidBodyNode

LaneIndex = Tuple[str, str, int]


class TrafficObject(BaseStaticObject):
    """
    Common interface for objects that appear on the road, beside vehicles.
    """
    NAME = BodyName.Traffic_object
    COLLISION_GROUP = CollisionGroup.TrafficObject

    COST_ONCE = True  # cost will give at the first time

    def __init__(self, lane, longitude: float, lateral: float, random_seed):
        """
        :param lane: the lane to spawn object
        :param longitude: use to calculate cartesian position of object in the surface
        :param lateral: use to calculate the angle from positive direction of horizontal axis
        """
        position = lane.position(longitude, lateral)
        heading_theta = lane.heading_theta_at(longitude)
        assert self.NAME is not None, "Assign a name for this class for finding it easily"
        super(TrafficObject, self).__init__(lane, position, heading_theta, random_seed)
        self.crashed = False


class TrafficCone(TrafficObject):
    """Placed near the construction section to indicate that traffic is prohibited"""

    RADIUS = 0.25
    HEIGHT = 1.2
    MASS = 1

    def __init__(self, lane, longitude: float, lateral: float, static: bool = False, random_seed=None):
        super(TrafficCone, self).__init__(lane, longitude, lateral, random_seed)
        self.add_body(BaseRigidBodyNode(self.name, self.NAME))
        self.body.addShape(BulletCylinderShape(self.RADIUS, self.HEIGHT))
        self.body.setIntoCollideMask(self.COLLISION_GROUP)
        self.set_static(static)
        if self.render:
            model = self.loader.loadModel(AssetLoader.file_path("models", "traffic_cone", "scene.gltf"))
            model.setScale(0.02)
            model.setPos(0, 0, -self.HEIGHT / 2)
            model.reparentTo(self.origin)

    @property
    def top_down_length(self):
        return self.RADIUS * 4

    @property
    def top_down_width(self):
        return self.RADIUS * 4

    @property
    def top_down_color(self):
        return 235, 84, 42


class TrafficWarning(TrafficObject):
    """Placed behind the vehicle when it breaks down"""

    HEIGHT = 1.2
    MASS = 1
    RADIUS = 0.5

    def __init__(self, lane, longitude: float, lateral: float, static: bool = False, random_seed=None):
        super(TrafficWarning, self).__init__(lane, longitude, lateral, random_seed)
        self.add_body(BaseRigidBodyNode(self.name, self.NAME))
        self.body.addShape(BulletCylinderShape(self.RADIUS, self.HEIGHT))
        self.body.setIntoCollideMask(self.COLLISION_GROUP)
        self.set_static(static)
        if self.render:
            model = self.loader.loadModel(AssetLoader.file_path("models", "warning", "warning.gltf"))
            model.setScale(0.02)
            model.setH(-90)
            model.setPos(0, 0, -self.HEIGHT / 2)
            model.reparentTo(self.origin)

    @property
    def top_down_length(self):
        return self.RADIUS * 2

    @property
    def top_down_width(self):
        return self.RADIUS * 2


class TrafficBarrier(TrafficObject):
    """A barrier"""

    HEIGHT = 2.0
    MASS = 10
    LENGTH = 2.0
    WIDTH = 0.3

    def __init__(self, lane, longitude: float, lateral: float, static: bool = False, random_seed=None):
        super(TrafficBarrier, self).__init__(lane, longitude, lateral, random_seed)
        self.add_body(BaseRigidBodyNode(self.name, self.NAME))
        self.body.addShape(BulletBoxShape((self.WIDTH / 2, self.LENGTH / 2, self.HEIGHT / 2)))
        self.body.setIntoCollideMask(self.COLLISION_GROUP)
        self.set_static(static)
        if self.render:
            model = self.loader.loadModel(AssetLoader.file_path("models", "barrier", "scene.gltf"))
            model.setH(-90)
            model.reparentTo(self.origin)

    @property
    def top_down_length(self):
        # reverse the direction
        return self.WIDTH

    @property
    def top_down_width(self):
        # reverse the direction
        return self.LENGTH
