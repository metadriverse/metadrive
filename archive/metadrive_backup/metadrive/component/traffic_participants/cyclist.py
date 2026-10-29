from panda3d.bullet import BulletBoxShape
from panda3d.core import LineSegs, NodePath

from metadrive.component.traffic_participants.base_traffic_participant import BaseTrafficParticipant
from metadrive.constants import CollisionGroup
from metadrive.constants import MetaDriveType, Semantics, get_color_palette
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.physics_node import BaseRigidBodyNode


class Cyclist(BaseTrafficParticipant):
    MASS = 80  # kg
    TYPE_NAME = MetaDriveType.CYCLIST
    COLLISION_MASK = CollisionGroup.TrafficParticipants
    SEMANTIC_LABEL = Semantics.BIKE.label
    MODEL = None

    HEIGHT = 1.75

    DEFAULT_LENGTH = 1.75  # meters
    DEFAULT_HEIGHT = 1.75  # meters
    DEFAULT_WIDTH = 0.4  # meters

    @property
    def LENGTH(self):
        return self.DEFAULT_LENGTH

    @property
    def HEIGHT(self):
        return self.DEFAULT_HEIGHT

    @property
    def WIDTH(self):
        return self.DEFAULT_WIDTH

    def __init__(self, position, heading_theta, random_seed, name=None, **kwargs):
        super(Cyclist, self).__init__(position, heading_theta, random_seed, name=name)
        self.set_metadrive_type(self.TYPE_NAME)
        n = BaseRigidBodyNode(self.name, self.TYPE_NAME)
        self.add_body(n)

        self.body.addShape(BulletBoxShape((self.WIDTH / 2, self.LENGTH / 2, self.HEIGHT / 2)))
        if self.render:
            if Cyclist.MODEL is None:
                model = self.loader.loadModel(AssetLoader.file_path("models", "bicycle", "scene.gltf"))
                model.setScale(0.15)
                model.setPos(0, 0, -0.3)
                Cyclist.MODEL = model
            Cyclist.MODEL.instanceTo(self.origin)

    def set_velocity(self, direction, value=None, in_local_frame=False):
        super(Cyclist, self).set_velocity(direction, value, in_local_frame)
        self.standup()

    def get_state(self):
        state = super(Cyclist, self).get_state()
        state.update({
            "length": self.LENGTH,
            "width": self.WIDTH,
            "height": self.HEIGHT,
        })
        return state


class CyclistBoundingBox(BaseTrafficParticipant):
    MASS = 80  # kg
    TYPE_NAME = MetaDriveType.CYCLIST
    COLLISION_MASK = CollisionGroup.TrafficParticipants
    SEMANTIC_LABEL = Semantics.BIKE.label

    # for random color choosing
    MATERIAL_COLOR_COEFF = 1.6  # to resist other factors, since other setting may make color dark
    MATERIAL_METAL_COEFF = 0.1  # 0-1
    MATERIAL_ROUGHNESS = 0.8  # smaller to make it more smooth, and reflect more light
    MATERIAL_SHININESS = 128  # 0-128 smaller to make it more smooth, and reflect more light
    MATERIAL_SPECULAR_COLOR = (3, 3, 3, 3)

    def __init__(self, position, heading_theta, random_seed, name=None, **kwargs):
        config = {"width": kwargs["width"], "length": kwargs["length"], "height": kwargs["height"]}
        # config = {"width": kwargs["length"], "length": kwargs["width"], "height": kwargs["height"]}
        super(CyclistBoundingBox, self).__init__(position, heading_theta, random_seed, name=name, config=config)
        self.set_metadrive_type(self.TYPE_NAME)
        n = BaseRigidBodyNode(self.name, self.TYPE_NAME)
        self.add_body(n)

        self.body.addShape(BulletBoxShape((self.WIDTH / 2, self.LENGTH / 2, self.HEIGHT / 2)))
        if self.render:
            model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
            model.setScale((self.WIDTH, self.LENGTH, self.HEIGHT))
            model.setTwoSided(False)
            self._instance = model.instanceTo(self.origin)

            # Add some color to help debug
            from panda3d.core import Material, LVecBase4

            show_contour = self.config["show_contour"] if "show_contour" in self.config else False
            if show_contour:
                # ========== Draw the contour of the bounding box ==========
                # Draw the bottom of the car first
                line_seg = LineSegs("bounding_box_contour1")
                zoffset = model.getZ()
                line_seg.setThickness(2)
                line_color = [0.0, 0.0, 0.0]
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
            rand_c = (1.0, 0.0, 0.0)
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
        super(CyclistBoundingBox, self).reset(position, heading_theta, random_seed, name, *args, **kwargs)
        config = {"width": kwargs["width"], "length": kwargs["length"], "height": kwargs["height"]}
        self.update_config(config)
        if self._instance is not None:
            self._instance.detachNode()
        if self.render:
            model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
            model.setScale((self.WIDTH, self.LENGTH, self.HEIGHT))
            model.setTwoSided(False)
            self._instance = model.instanceTo(self.origin)

            # Add some color to help debug
            from panda3d.core import Material, LVecBase4
            color = list(COLOR_PALETTE)
            color.remove(color[2])  # Remove the green and leave it for special vehicle
            idx = 0
            rand_c = color[idx]
            rand_c = (1.0, 0.0, 0.0)
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

    def set_velocity(self, direction, value=None, in_local_frame=False):
        super(CyclistBoundingBox, self).set_velocity(direction, value, in_local_frame)
        self.standup()

    @property
    def WIDTH(self):
        # return self.config["width"]
        return self.config["length"]

    @property
    def HEIGHT(self):
        return self.config["height"]

    @property
    def LENGTH(self):
        # return self.config["length"]
        return self.config["width"]
