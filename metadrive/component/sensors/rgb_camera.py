import panda3d.core as p3d
from direct.filter.FilterManager import FilterManager
from simplepbr import _load_shader_str

from metadrive.component.sensors.base_camera import BaseCamera
from metadrive.constants import CamMask


class RGBCamera(BaseCamera):
    # shape(dim_1, dim_2)
    BUFFER_W = 84  # dim 1
    BUFFER_H = 84  # dim 2
    CAM_MASK = CamMask.RgbCam
    PBR_ADAPT = False

    frame_buffer_rgb_bits = (8, 8, 8, 0)

    def __init__(self, width, height, engine, *, cuda=False):
        self.BUFFER_W, self.BUFFER_H = width, height
        super(RGBCamera, self).__init__(engine, cuda)
        cam = self.get_cam()
        lens = self.get_lens()
        # cam.lookAt(0, 2.4, 1.3)
        cam.lookAt(0, 10.4, 1.6)
        lens.setFov(60)

    def _setup_effect(self):
        """
        Setup simple PBR effect
        Returns: None

        """
        self.scene_tex = None
        self.manager = FilterManager(self.buffer, self.cam)
        fbprops = p3d.FrameBufferProperties()
        fbprops.float_color = True
        fbprops.set_rgba_bits(16, 16, 16, 16)
        fbprops.set_depth_bits(24)
        fbprops.set_multisamples(self.engine.pbrpipe.msaa_samples)
        self.scene_tex = p3d.Texture()
        self.scene_tex.set_format(p3d.Texture.F_rgba16)
        self.scene_tex.set_component_type(p3d.Texture.T_float)
        self.tonemap_quad = self.manager.render_scene_into(colortex=self.scene_tex, fbprops=fbprops)
        #
        defines = {}
        #
        post_vert_str = _load_shader_str('post.vert', defines)
        post_frag_str = _load_shader_str('tonemap.frag', defines)
        tonemap_shader = p3d.Shader.make(
            p3d.Shader.SL_GLSL,
            vertex=post_vert_str,
            fragment=post_frag_str,
        )
        self.tonemap_quad.set_shader(tonemap_shader)
        self.tonemap_quad.set_shader_input('tex', self.scene_tex)
        self.tonemap_quad.set_shader_input('exposure', 1.0)