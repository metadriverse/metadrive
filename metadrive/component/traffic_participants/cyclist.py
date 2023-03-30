from metadrive.component.traffic_participants.base_traffic_participant import BaseTrafficParticipant
from typing import Tuple

from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletCylinderShape

from metadrive.component.static_object.base_static_object import BaseStaticObject
from metadrive.constants import BodyName
from metadrive.constants import CollisionGroup
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.physics_node import BaseRigidBodyNode


class Cyclist(BaseTrafficParticipant):
    MASS = 80  # kg
    TYPE_NAME = BodyName.Cyclist
    COLLISION_MASK = CollisionGroup.TrafficParticipants

    MODEL = None

    HEIGHT = 1.75

    def __init__(self, position, heading_theta, random_seed):
        super(Cyclist, self).__init__(position, heading_theta, random_seed)

        n = BaseRigidBodyNode(self.name, self.TYPE_NAME)
        self.add_body(n)

        self.body.addShape(BulletBoxShape((self.LENGTH / 2, self.WIDTH / 2, self.HEIGHT / 2)))
        if self.render:
            if Cyclist.MODEL is None:
                model = self.loader.loadModel(AssetLoader.file_path("models", "bicycle", "scene.gltf"))
                model.setScale(0.15)
                model.setPos(0, 0, -0.3)
                Cyclist.MODEL = model
            Cyclist.MODEL.instanceTo(self.origin)

    @property
    def WIDTH(self):
        return 0.4

    @property
    def LENGTH(self):
        return 1.75
