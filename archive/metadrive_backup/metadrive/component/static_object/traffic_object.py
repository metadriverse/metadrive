import logging
from typing import Tuple

from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletCylinderShape

from metadrive.component.static_object.base_static_object import BaseStaticObject
from metadrive.constants import CollisionGroup
from metadrive.constants import MetaDriveType, Semantics
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.physics_node import BaseRigidBodyNode

LaneIndex = Tuple[str, str, int]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficObject(BaseStaticObject):
    """
    Common interface for objects that appear on the road, beside vehicles.
    """
    CLASS_NAME = MetaDriveType.TRAFFIC_OBJECT
    COLLISION_MASK = CollisionGroup.TrafficObject
    SEMANTIC_LABEL = Semantics.FENCE.label
    MODEL = None

    COST_ONCE = True  # cost will give at the first time

    def __init__(self, position, heading_theta, lane=None, random_seed=None, name=None):
        """
        :param lane: the lane to spawn object
        """
        assert self.CLASS_NAME is not None, "Assign a name for this class for finding it easily"
        super(TrafficObject, self).__init__(position, heading_theta, lane, random_seed, name=name)
        self.set_metadrive_type(self.CLASS_NAME)
        self.crashed = False

    def reset(self, position, heading_theta, lane=None, random_seed=None, name=None, *args, **kwargs):
        self.crashed = False
        super(TrafficObject, self).reset(position, heading_theta, lane, random_seed, name, *args, **kwargs)


class TrafficCone(TrafficObject):
    """Placed near the construction section to indicate that traffic is prohibited"""

    RADIUS = 0.2
    HEIGHT = 2
    MASS = 2
    CLASS_NAME = MetaDriveType.TRAFFIC_CONE

    def __init__(self, position, heading_theta, lane=None, static: bool = False, random_seed=None, name=None):
        super(TrafficCone, self).__init__(position, heading_theta, lane, random_seed, name)

        n = BaseRigidBodyNode(self.name, self.CLASS_NAME)
        self.add_body(n)

        self.body.addShape(BulletCylinderShape(self.RADIUS, self.HEIGHT))
        self.set_static(static)
        if self.render:
            if TrafficCone.MODEL is None:
                model = self.loader.loadModel(AssetLoader.file_path("models", "traffic_cone", "scene.gltf"))
                # model.node().setTag("semantic", "vehicle")
                model.setScale(0.02, 0.02, 0.025)
                model.setPos(0, 0, -self.HEIGHT / 2)
                TrafficCone.MODEL = model
            TrafficCone.MODEL.instanceTo(self.origin)

    @property
    def top_down_length(self):
        return self.RADIUS * 4

    @property
    def top_down_width(self):
        return self.RADIUS * 4

    @property
    def top_down_color(self):
        return 235, 84, 42

    @property
    def LENGTH(self):
        return self.RADIUS

    @property
    def WIDTH(self):
        return self.RADIUS


class TrafficWarning(TrafficObject):
    """Placed behind the vehicle when it breaks down"""

    HEIGHT = 1.2
    MASS = 1
    RADIUS = 0.5

    def __init__(self, position, heading_theta, lane=None, static: bool = False, random_seed=None, name=None):
        super(TrafficWarning, self).__init__(position, heading_theta, lane, random_seed, name)

        n = BaseRigidBodyNode(self.name, self.CLASS_NAME)
        self.add_body(n)

        self.body.addShape(BulletCylinderShape(self.RADIUS, self.HEIGHT))
        self.set_static(static)
        if self.render:
            if TrafficWarning.MODEL is None:
                model = self.loader.loadModel(AssetLoader.file_path("models", "warning", "warning.gltf"))
                model.setScale(0.02)
                model.setH(-90)
                model.setPos(0, 0, -self.HEIGHT / 2 - 0.1)
                TrafficWarning.MODEL = model
            TrafficWarning.MODEL.instanceTo(self.origin)

    @property
    def top_down_length(self):
        return self.RADIUS * 2

    @property
    def top_down_width(self):
        return self.RADIUS * 2

    @property
    def LENGTH(self):
        return self.RADIUS

    @property
    def WIDTH(self):
        return self.RADIUS


class TrafficBarrier(TrafficObject):
    """A barrier"""

    HEIGHT = 2.0
    MASS = 10
    CLASS_NAME = MetaDriveType.TRAFFIC_BARRIER

    def __init__(self, position, heading_theta, lane=None, static: bool = False, random_seed=None, name=None):
        super(TrafficBarrier, self).__init__(position, heading_theta, lane, random_seed, name)
        n = BaseRigidBodyNode(self.name, self.CLASS_NAME)
        self.add_body(n)

        self.body.addShape(BulletBoxShape((self.WIDTH / 2, self.LENGTH / 2, self.height / 2)))
        self.set_static(static)
        if self.render:
            if TrafficBarrier.MODEL is None:
                model = self.loader.loadModel(AssetLoader.file_path("models", "barrier", "scene.gltf"))
                model.setH(-90)
                model.setPos(0, 0, -1.05)
                model.setScale(0.7)
                TrafficBarrier.MODEL = model
            TrafficBarrier.MODEL.instanceTo(self.origin)

    @property
    def LENGTH(self):
        return 2.0

    @property
    def WIDTH(self):
        return 0.3

    @property
    def width(self):
        logger.warning("This API will be deprecated, Please use {}.WIDTH instead".format(self.class_name))
        return self.WIDTH

    @property
    def length(self):
        logger.warning("This API will be deprecated, Please use {}.LENGTH instead".format(self.class_name))
        return self.LENGTH

    @property
    def height(self):
        return self.HEIGHT

    @property
    def top_down_length(self):
        # reverse the direction
        return self.WIDTH * 2

    @property
    def top_down_width(self):
        # reverse the direction
        return self.LENGTH
