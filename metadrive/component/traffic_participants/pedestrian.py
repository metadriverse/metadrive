from metadrive.component.traffic_participants.base_traffic_participant import BaseTrafficParticipant
from typing import Tuple

from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletCylinderShape

from metadrive.component.static_object.base_static_object import BaseStaticObject
from direct.actor.Actor import Actor
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

    _MODEL = None

    def __init__(self, position, heading_theta, random_seed):
        super(Pedestrian, self).__init__(position, heading_theta, random_seed)

        n = BaseRigidBodyNode(self.name, self.NAME)
        self.add_body(n)
        self._node_path_list.append(n)

        self.body.addShape(BulletCylinderShape(self.RADIUS, self.HEIGHT))
        self.body.setIntoCollideMask(self.COLLISION_GROUP)
        self.set_static(True)
        self.animation_controller = None
        if self.render:
            # if Pedestrian._MODEL is None:
            model = Actor(AssetLoader.file_path("models", "pedestrian", "scene.gltf"))
            # model: Actor = self._MODEL
            model.setScale(0.01)
            model.setPos(0, 0, -self.HEIGHT / 2)
            model.reparentTo(self.origin)
            # self.animation_controller = model.get_anim_control("Take 001")
            # self.animation_controller.setPlayRate(1)
            # self.animation_controller.loop("Take 001")
