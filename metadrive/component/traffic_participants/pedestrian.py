from typing import Dict

from direct.actor.Actor import Actor
from panda3d.bullet import BulletCylinderShape, BulletBoxShape
from panda3d.core import LVecBase4
from panda3d.core import LVector3
from panda3d.core import LineSegs, NodePath

from metadrive.component.traffic_participants.base_traffic_participant import BaseTrafficParticipant
from metadrive.constants import MetaDriveType, Semantics, get_color_palette
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.physics_node import BaseRigidBodyNode
from metadrive.utils.math import norm


class Pedestrian(BaseTrafficParticipant):
    MASS = 70  # kg
    TYPE_NAME = MetaDriveType.PEDESTRIAN
    SEMANTIC_LABEL = Semantics.PEDESTRIAN.label
    RADIUS = 0.35
    HEIGHT = 1.75

    _MODEL = {}

    # SPEED_LIST = [0.6, 1.2, 2.2] Too much speed choice jeopardise the performance
    SPEED_LIST = [0.4, 1.2]

    def __init__(self, position, heading_theta, random_seed=None, name=None, *args, **kwargs):
        super(Pedestrian, self).__init__(position, heading_theta, random_seed, name=name)
        self.set_metadrive_type(self.TYPE_NAME)
        # self.origin.setDepthOffset(1)
        n = BaseRigidBodyNode(self.name, self.TYPE_NAME)
        self.add_body(n)
        self.body.addShape(BulletCylinderShape(self.RADIUS, self.HEIGHT))
        # self.set_static(True)
        self.animation_controller = None
        self.current_speed_model = self.SPEED_LIST[0]
        self._instance = None
        if self.render:
            if len(Pedestrian._MODEL) == 0:
                self.init_pedestrian_model()
            self._instance = Pedestrian._MODEL[self.current_speed_model].instanceTo(self.origin)
            self.show_coordinates()

    def reset(self, position, heading_theta: float = 0., random_seed=None, name=None, *args, **kwargs):
        super(Pedestrian, self).reset(position, heading_theta, random_seed, name, *args, **kwargs)
        self.current_speed_model = self.SPEED_LIST[0]
        if self._instance is not None:
            self._instance.detachNode()
        if self.render:
            self._instance = Pedestrian._MODEL[self.current_speed_model].instanceTo(self.origin)

    @classmethod
    def init_pedestrian_model(cls):
        for idx, speed in enumerate(cls.SPEED_LIST):
            model = Actor(AssetLoader.file_path("models", "pedestrian", "scene.gltf"))
            # model: Actor = self._MODEL
            model.setScale(0.01)
            model.setH(model.getH() + 90)
            model.setPos(0, 0, -cls.HEIGHT / 2)
            Pedestrian._MODEL[speed] = model
            if idx == 0:
                animation_controller = model.get_anim_control("Take 001")
                animation_controller.setPlayRate(0)
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
            direction = ret
        speed = (norm(direction[0], direction[1]) + 1e-6)
        if value is not None:
            norm_ratio = value / speed
        else:
            norm_ratio = 1

        if self.render:
            speed_model_index = self.get_speed_model(target_speed=speed if value is None else value)
            if speed_model_index != self.current_speed_model:
                self._instance.detachNode()
                self._instance = Pedestrian._MODEL[speed_model_index].instanceTo(self.origin)
                self.current_speed_model = speed_model_index
            self.show_coordinates()

        self._body.setLinearVelocity(
            LVector3(direction[0] * norm_ratio, direction[1] * norm_ratio,
                     self._body.getLinearVelocity()[-1])
        )
        self.standup()

    @staticmethod
    def get_speed_model(target_speed):
        for speed in Pedestrian.SPEED_LIST:
            if target_speed < speed:
                return speed
        return Pedestrian.SPEED_LIST[-1]

    @property
    def LENGTH(self):
        return self.RADIUS

    @property
    def WIDTH(self):
        return self.RADIUS

    @property
    def top_down_width(self):
        return self.RADIUS * 2

    @property
    def top_down_length(self):
        return self.RADIUS * 2

    def get_state(self) -> Dict:
        state = super(Pedestrian, self).get_state()
        state.update(
            {
                "length": self.RADIUS * 2,
                "width": self.RADIUS * 2,
                "height": self.HEIGHT,
                "radius": self.RADIUS,
            }
        )
        return state


class PedestrianBoundingBox(BaseTrafficParticipant):
    MASS = 70  # kg
    TYPE_NAME = MetaDriveType.PEDESTRIAN
    SEMANTIC_LABEL = Semantics.PEDESTRIAN.label

    # for random color choosing
    MATERIAL_COLOR_COEFF = 1.6  # to resist other factors, since other setting may make color dark
    MATERIAL_METAL_COEFF = 0.1  # 0-1
    MATERIAL_ROUGHNESS = 0.8  # smaller to make it more smooth, and reflect more light
    MATERIAL_SHININESS = 128  # 0-128 smaller to make it more smooth, and reflect more light
    MATERIAL_SPECULAR_COLOR = (3, 3, 3, 3)

    def __init__(self, position, heading_theta, width, length, height, random_seed=None, name=None):
        config = {}
        config["width"] = width
        config["length"] = length
        config["height"] = height

        super(PedestrianBoundingBox, self).__init__(position, heading_theta, random_seed, name=name, config=config)
        self.set_metadrive_type(self.TYPE_NAME)
        n = BaseRigidBodyNode(self.name, self.TYPE_NAME)
        self.add_body(n)

        # PZH: Use BoxShape instead of CylinderShape
        # self.body.addShape(BulletCylinderShape(self.RADIUS, self.HEIGHT))
        self.body.addShape(BulletBoxShape(LVector3(width / 2, length / 2, height / 2)))

        self._instance = None
        if self.render:
            # PZH: Load a box model and resize it to the vehicle size
            model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
            model.setScale((self.WIDTH, self.LENGTH, self.HEIGHT))
            model.setTwoSided(False)
            self._instance = model.instanceTo(self.origin)

            # Add some color to help debug
            from panda3d.core import Material

            show_contour = self.config["show_contour"] if "show_contour" in self.config else False
            if show_contour:
                # ========== Draw the contour of the bounding box ==========
                # Draw the bottom of the car first
                line_seg = LineSegs("bounding_box_contour1")
                zoffset = model.getZ()
                line_seg.setThickness(2)
                line_color = [0.0, 0.0, 1.0]
                out_offset = 0.02
                w = self.WIDTH / 2 + out_offset
                l = self.LENGTH / 2 + out_offset
                h = self.HEIGHT / 2 + out_offset
                line_seg.moveTo(w, l, h + zoffset)
                line_seg.drawTo(-w, l, h + zoffset)
                line_seg.drawTo(-w, l, -h + zoffset)
                line_seg.drawTo(w, l, -h + zoffset)
                line_seg.drawTo(w, l, h + zoffset)

                # draw cross line
                line_seg.moveTo(w, l, h + zoffset)
                line_seg.drawTo(w, -l, -h + zoffset)
                line_seg.moveTo(w, -l, h + zoffset)
                line_seg.drawTo(w, l, -h + zoffset)

                line_seg.moveTo(w, -l, h + zoffset)
                line_seg.drawTo(-w, -l, h + zoffset)
                line_seg.drawTo(-w, -l, -h + zoffset)
                line_seg.drawTo(w, -l, -h + zoffset)
                line_seg.drawTo(w, -l, h + zoffset)

                # draw vertical & horizontal line
                line_seg.moveTo(-w, l, 0 + zoffset)
                line_seg.drawTo(-w, -l, 0 + zoffset)
                line_seg.moveTo(-w, 0, h + zoffset)
                line_seg.drawTo(-w, 0, -h + zoffset)

                line_seg.moveTo(w, l, h + zoffset)
                line_seg.drawTo(w, -l, h + zoffset)
                line_seg.moveTo(-w, l, h + zoffset)
                line_seg.drawTo(-w, -l, h + zoffset)
                line_seg.moveTo(-w, l, -h + zoffset)
                line_seg.drawTo(-w, -l, -h + zoffset)
                line_seg.moveTo(w, l, -h + zoffset)
                line_seg.drawTo(w, -l, -h + zoffset)
                line_np = NodePath(line_seg.create(True))
                line_material = Material()
                line_material.setBaseColor(LVecBase4(*line_color[:3], 1))
                line_np.setMaterial(line_material, True)
                line_np.reparentTo(self.origin)

            color = get_color_palette()
            color.remove(color[2])  # Remove the green and leave it for special vehicle
            idx = 0
            rand_c = color[idx]
            rand_c = (0.0, 1.0, 0.0)
            self._panda_color = rand_c
            material = Material()
            material.setBaseColor(
                (
                    self.panda_color[0] * self.MATERIAL_COLOR_COEFF, self.panda_color[1] * self.MATERIAL_COLOR_COEFF,
                    self.panda_color[2] * self.MATERIAL_COLOR_COEFF, 0.
                )
            )
            material.setMetallic(self.MATERIAL_METAL_COEFF)
            material.setSpecular(self.MATERIAL_SPECULAR_COLOR)
            material.setRefractiveIndex(1.5)
            material.setRoughness(self.MATERIAL_ROUGHNESS)
            material.setShininess(self.MATERIAL_SHININESS)
            material.setTwoside(False)
            self.origin.setMaterial(material, True)

    def reset(self, position, heading_theta: float = 0., random_seed=None, name=None, *args, **kwargs):
        super(PedestrianBoundingBox, self).reset(position, heading_theta, random_seed, name, *args, **kwargs)
        config = {"width": kwargs["width"], "length": kwargs["length"], "height": kwargs["height"]}
        self.update_config(config)
        if self._instance is not None:
            self._instance.detachNode()
        if self.render:
            model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
            model.setScale((self.WIDTH, self.LENGTH, self.HEIGHT))
            model.setTwoSided(False)
            self._instance = model.instanceTo(self.origin)

    def set_velocity(self, direction: list, value=None, in_local_frame=False):
        self.set_roll(0)
        self.set_pitch(0)
        if in_local_frame:
            from metadrive.engine.engine_utils import get_engine
            engine = get_engine()
            direction = LVector3(*direction, 0.)
            direction[1] *= -1
            ret = engine.worldNP.getRelativeVector(self.origin, direction)
            direction = ret
        speed = (norm(direction[0], direction[1]) + 1e-6)
        if value is not None:
            norm_ratio = value / speed
        else:
            norm_ratio = 1

        self._body.setLinearVelocity(
            LVector3(direction[0] * norm_ratio, direction[1] * norm_ratio,
                     self._body.getLinearVelocity()[-1])
        )
        self.standup()

    @property
    def HEIGHT(self):
        return self.config["height"]

    @property
    def LENGTH(self):
        return self.config["length"]

    @property
    def WIDTH(self):
        return self.config["width"]
