# import numpy
import math

from panda3d.bullet import BulletRigidBodyNode, BulletPlaneShape
from panda3d.core import Vec3, CardMaker, LQuaternionf, TextureStage, Texture, SamplerState

from metadrive.base_class.base_object import BaseObject
from metadrive.constants import BodyName, CamMask, CollisionGroup
from metadrive.engine.asset_loader import AssetLoader


class Terrain(BaseObject):
    COLLISION_MASK = CollisionGroup.Terrain
    HEIGHT = 0.0

    def __init__(self, show_terrain):
        super(Terrain, self).__init__(random_seed=0)
        shape = BulletPlaneShape(Vec3(0, 0, 1), 0)
        node = BulletRigidBodyNode(BodyName.Ground)
        node.setFriction(.9)
        node.addShape(shape)

        node.setIntoCollideMask(self.COLLISION_MASK)
        self.dynamic_nodes.append(node)

        self.origin.attachNewNode(node)
        if self.render and show_terrain:
            self.origin.hide(CamMask.MiniMap | CamMask.Shadow | CamMask.DepthCam | CamMask.ScreenshotCam)
            # self.terrain_normal = self.loader.loadTexture(
            #     AssetLoader.file_path( "textures", "grass2", "normal.jpg")
            # )
            self.terrain_texture = self.loader.loadTexture(AssetLoader.file_path("textures", "ground.png"))
            self.terrain_texture.setWrapU(Texture.WM_repeat)
            self.terrain_texture.setWrapV(Texture.WM_repeat)
            self.ts_color = TextureStage("color")
            self.ts_normal = TextureStage("normal")
            self.ts_normal.set_mode(TextureStage.M_normal)
            self.set_position((0, 0), self.HEIGHT)
            cm = CardMaker('card')
            scale = 20000
            cm.setUvRange((0, 0), (scale / 10, scale / 10))
            card = self.origin.attachNewNode(cm.generate())
            # scale = 1 if self.use_hollow else 20000
            card.set_scale(scale)
            card.setPos(-scale / 2, -scale / 2, -0.1)
            card.setZ(-.05)
            card.setTexture(self.ts_color, self.terrain_texture)
            # card.setTexture(self.ts_normal, self.terrain_normal)
            self.terrain_texture.setMinfilter(SamplerState.FT_linear_mipmap_linear)
            self.terrain_texture.setAnisotropicDegree(8)
            card.setQuat(LQuaternionf(math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0, 0))
