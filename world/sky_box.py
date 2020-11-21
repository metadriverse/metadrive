import os
from panda3d.core import SamplerState, Shader
from utils.visualization_loader import VisLoader
from utils.element import DynamicElement


class SkyBox(DynamicElement):
    """
    SkyBox is only related to render
    """
    ROTATION_MAX = 5000

    def __init__(self):
        super(SkyBox, self).__init__()
        if not self.render:
            return
        skybox = self.loader.loadModel(os.path.join(VisLoader.path, "models/skybox.bam"))
        from pg_config.cam_mask import CamMask
        skybox.hide(CamMask.MiniMap)
        # skybox.setScale(512)
        # skybox_texture = self.loader.loadTexture(os.path.join(VisLoader.path, 'textures/skybox.jpg'))
        # # skybox.setBin(
        # #     os.path.join(self.bullet_path, 'textures/s1/background#.jpg')
        # #     , 1)
        # # skybox.setDepthWrite(0)
        #
        # skybox.setLightOff()
        #
        # ts = TextureStage('ts')
        # ts.setMode(TextureStage.MReplace)
        #
        # # skybox.setTexGen(ts, TexGenAttrib.MWorldNormal)
        # skybox.setTexture(ts, skybox_texture)
        #
        # # skybox.setBin(os.path.join(self.bullet_path, 'textures/s1/background'), 1)
        # # skybox.setScale(20000)
        # # skybox.setZ(-2450)
        # self.node_path = skybox
        # # skybox.reparent_to(self.render)
        # # skybox.hide(DrawMask(self.MINIMAP_MASK))

        # skybox = self.loader.loadModel(os.path.join(self.bullet_path, "models/skybox.bam"))
        skybox.set_scale(20000)

        skybox_texture = self.loader.loadTexture(os.path.join(VisLoader.path, "textures/skybox.jpg"))
        skybox_texture.set_minfilter(SamplerState.FT_linear)
        skybox_texture.set_magfilter(SamplerState.FT_linear)
        skybox_texture.set_wrap_u(SamplerState.WM_repeat)
        skybox_texture.set_wrap_v(SamplerState.WM_mirror)
        skybox_texture.set_anisotropic_degree(16)
        skybox.set_texture(skybox_texture)

        skybox_shader = Shader.load(
            Shader.SL_GLSL, os.path.join(VisLoader.path, "models/skybox.vert.glsl"),
            os.path.join(VisLoader.path, "models/skybox.frag.glsl")
        )
        skybox.set_shader(skybox_shader)
        self.node_path = skybox
        skybox.setZ(-4400)
        skybox.setH(30)
        self._accumulate = 0
        self.f = 1

    def step(self):
        if not self.render:
            return
        if self._accumulate >= self.ROTATION_MAX:
            self.f *= -1
            self._accumulate = 0
        self._accumulate += 1
        factor = self.f * (1 - abs(self._accumulate - self.ROTATION_MAX / 2) * 2 / self.ROTATION_MAX)
        self.node_path.setH(self.node_path.getH() + factor * 0.0015)
