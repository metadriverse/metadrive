from metadrive.component.traffic_participants.base_traffic_participant import BaseTrafficParticipant
from typing import Tuple

from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletCylinderShape

from metadrive.component.static_object.base_static_object import BaseStaticObject
from metadrive.constants import BodyName
from metadrive.constants import CollisionGroup
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.physics_node import BaseRigidBodyNode


class Pedestrian(BaseTrafficParticipant):
    MASS = 70  # kg
    NAME = BodyName.Pedestrian
    COLLISION_GROUP = CollisionGroup.TrafficParticipants

    RADIUS = 0.3
    HEIGHT = 1.75

    def __init__(self, position, heading_theta, random_seed):
        super(Pedestrian, self).__init__(position, heading_theta, random_seed)

        n = BaseRigidBodyNode(self.name, self.NAME)
        self.add_body(n)
        self._node_path_list.append(n)

        self.body.addShape(BulletCylinderShape(self.RADIUS, self.HEIGHT))
        self.body.setIntoCollideMask(self.COLLISION_GROUP)
        self.set_static(True)
        if self.render:
            model = self.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
            model.setScale(self.RADIUS, self.RADIUS, self.HEIGHT)
            model.setPos(0, 0, -self.HEIGHT / 2)
            model.reparentTo(self.origin)
