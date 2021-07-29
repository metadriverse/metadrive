from panda3d.core import SamplerState, Shader, NodePath, ConfigVariableString

from pgdrive.constants import CamMask
from pgdrive.engine.asset_loader import AssetLoader
from pgdrive.utils.object import Object
from pgdrive.utils.utils import is_mac


class SkyBox(Object):
    """
    SkyBox is only related to render
    """
    ROTATION_MAX = 5000

    def __init__(self, pure_background: bool = False):
        super(SkyBox, self).__init__(random_seed=0)
        self._accumulate = 0
        self.f = 1
        if not self.render or pure_background:
            self.node_path = NodePath("pure_background")
            return
        skybox = self.loader.loadModel(AssetLoader.file_path("models", "skybox.bam"))

        skybox.hide(CamMask.MiniMap | CamMask.RgbCam | CamMask.Shadow | CamMask.ScreenshotCam)
        skybox.set_scale(20000)

        skybox_texture = self.loader.loadTexture(AssetLoader.file_path("textures", "skybox.jpg"))
        skybox_texture.set_minfilter(SamplerState.FT_linear)
        skybox_texture.set_magfilter(SamplerState.FT_linear)
        skybox_texture.set_wrap_u(SamplerState.WM_repeat)
        skybox_texture.set_wrap_v(SamplerState.WM_mirror)
        skybox_texture.set_anisotropic_degree(16)
        skybox.set_texture(skybox_texture)

        gles = ConfigVariableString("load-display").getValue()
        if gles == "pandagles2":
            skybox_shader = Shader.load(
                Shader.SL_GLSL, AssetLoader.file_path("shaders", "skybox_gles.vert.glsl"),
                AssetLoader.file_path("shaders", "skybox_gles.frag.glsl")
            )
        else:
            if is_mac():
                vert_file = "skybox_mac.vert.glsl"
                frag_file = "skybox_mac.frag.glsl"
            else:
                vert_file = "skybox.vert.glsl"
                frag_file = "skybox.frag.glsl"
            skybox_shader = Shader.load(
                Shader.SL_GLSL, AssetLoader.file_path("shaders", vert_file),
                AssetLoader.file_path("shaders", frag_file)
            )
        skybox.set_shader(skybox_shader)
        self.node_path = skybox
        skybox.setZ(-4400)
        skybox.setH(30)

    def step(self):
        if not self.render:
            return
        if self._accumulate >= self.ROTATION_MAX:
            self.f *= -1
            self._accumulate = 0
        self._accumulate += 1
        factor = self.f * (1 - abs(self._accumulate - self.ROTATION_MAX / 2) * 2 / self.ROTATION_MAX)
        self.node_path.setH(self.node_path.getH() + factor * 0.0035)
