from direct.actor.Actor import Actor
from panda3d.bullet import BulletCylinderShape
from panda3d.core import LVector3

from metadrive.component.traffic_participants.base_traffic_participant import BaseTrafficParticipant
from metadrive.constants import BodyName
from metadrive.constants import CollisionGroup
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.physics_node import BaseRigidBodyNode
from metadrive.utils.math_utils import norm


class Pedestrian(BaseTrafficParticipant):
    MASS = 70  # kg
    NAME = BodyName.Pedestrian
    COLLISION_GROUP = CollisionGroup.TrafficParticipants

    RADIUS = 0.35
    HEIGHT = 1.75

    _MODEL = {}

    SPEED_LIST = [0.6, 1.2, 2.2]

    def __init__(self, position, heading_theta, random_seed=None):
        super(Pedestrian, self).__init__(position, heading_theta, random_seed)

        n = BaseRigidBodyNode(self.name, self.NAME)
        self.add_body(n)
        self.body.addShape(BulletCylinderShape(self.RADIUS, self.HEIGHT))
        self.body.setIntoCollideMask(self.COLLISION_GROUP)
        # self.set_static(True)
        self.animation_controller = None
        self.current_speed_model = self.SPEED_LIST[0]
        if self.render:
            if len(Pedestrian._MODEL) == 0:
                self._init_pedestrian_model()
            Pedestrian._MODEL[self.current_speed_model].instanceTo(self.origin)
            self.show_coordinates()

    @classmethod
    def _init_pedestrian_model(cls):
        for idx, speed in enumerate(cls.SPEED_LIST):
            model = Actor(AssetLoader.file_path("models", "pedestrian", "scene.gltf"))
            # model: Actor = self._MODEL
            model.setScale(0.01)
            model.setH(model.getH() + 90)
            model.setPos(0, 0, -cls.HEIGHT / 2)
            Pedestrian._MODEL[speed] = model
            if idx == 0:
                animation_controller = model.get_anim_control("Take 001")
                animation_controller.setPlayRate(idx)
                animation_controller.pose(1)
            else:
                animation_controller = model.get_anim_control("Take 001")
                animation_controller.setPlayRate(speed / 2 + 0.2)
                animation_controller.loop("Take 001")

    def set_velocity(self, direction: list, value=None, in_local_frame=False):
        self.set_roll(0)
        self.set_pitch(0)
        if in_local_frame:
            from metadrive.engine.engine_utils import get_engine
            engine = get_engine()
            direction = LVector3(*direction, 0.)
            direction[1] *= -1
            ret = engine.worldNP.getRelativeVector(self.origin, direction)
            # no need for spinning 90 degree
            direction = ret
        speed = (norm(direction[0], direction[1]) + 1e-6)
        if value is not None:
            norm_ratio = value / speed
        else:
            norm_ratio = 1

        if self.render:
            speed_model_index = self.get_speed_model(target_speed=speed if value is None else value)
            if speed_model_index != self.current_speed_model:
                child = self.origin.getChildren()
                child.detach()
                Pedestrian._MODEL[speed_model_index].instanceTo(self.origin)
                self.current_speed_model = speed_model_index
            self.show_coordinates()

        self._body.setLinearVelocity(
            LVector3(direction[0] * norm_ratio, direction[1] * norm_ratio,
                     self._body.getLinearVelocity()[-1])
        )
        self.origin.setR(0)
        self.origin.setP(0)

    @staticmethod
    def get_speed_model(target_speed):
        for speed in Pedestrian.SPEED_LIST:
            if target_speed < speed:
                return speed
        return Pedestrian.SPEED_LIST[-1]
