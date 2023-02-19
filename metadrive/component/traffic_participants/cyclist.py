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
    NAME = BodyName.Cyclist
    COLLISION_GROUP = CollisionGroup.TrafficParticipants

    WIDTH = 0.4
    LENGTH = 1.75
    HEIGHT = 1.75

    def __init__(self, position, heading_theta, random_seed):
        super(Cyclist, self).__init__(position, heading_theta, random_seed)

        n = BaseRigidBodyNode(self.name, self.NAME)
        self.add_body(n)

        self.body.addShape(BulletBoxShape((self.WIDTH / 2, self.LENGTH / 2, self.HEIGHT / 2)))
        self.body.setIntoCollideMask(self.COLLISION_GROUP)
        self.set_static(True)
        if self.render:
            model = self.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
            # model.setH(-90)
            model.setScale(self.WIDTH, self.LENGTH, self.HEIGHT)
            model.reparentTo(self.origin)
