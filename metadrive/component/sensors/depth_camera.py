import numpy as np
from panda3d.core import Texture, Shader, NodePath, ShaderAttrib, LVector2
from panda3d.core import WindowProperties, FrameBufferProperties, GraphicsPipe, GraphicsOutput

from metadrive.component.sensors.base_camera import BaseCamera
from metadrive.constants import CamMask
from metadrive.engine.asset_loader import AssetLoader


class DepthCamera(BaseCamera):
    CAM_MASK = CamMask.DepthCam

    GROUND_HEIGHT = 0
    VIEW_GROUND = True
    GROUND = None
    GROUND_MODEL = None

    num_channels = 1

    def __init__(self, width, height, engine, *, cuda=False):
        self.BUFFER_W, self.BUFFER_H = width, height
        # factors of the log algorithm used to process distance to object
        self.log_b = np.log(16)
        self.log_base = np.log(5)
        self.log_base_div_b = self.log_base / self.log_b

        # create
        super(DepthCamera, self).__init__(engine, cuda)

        # for converting depth value to distance-based depth on CPU
        near = self.lens.getNear()
        far = self.lens.getFar()
        self.far_near_mul = near * far
        self.far_near_add = near + far
        self.far_near_minus = far - near

        if self.enable_cuda:
            self.engine.taskMgr.add(self._dispatch_compute)

    def _create_camera(self, pos, bkg_color):
        super(DepthCamera, self)._create_camera(pos, bkg_color)
        self.cam.node().set_scene(self.engine.render)
        self.buffer_display_region = self.cam.node().getDisplayRegion(0)
        self.buffer_display_region.set_sort(25)
        self.buffer_display_region.disable_clears()
        self.buffer_display_region.set_active(True)
        if self.enable_cuda:
            self.compute_node.set_shader_input("near", self.lens.getNear())
            self.compute_node.set_shader_input("far", self.lens.getFar())
            self.compute_node.set_shader_input("log_b", self.log_b)
            self.compute_node.set_shader_input("log_base_div_b", self.log_base_div_b)

    def _create_buffer(self, width, height, frame_buffer_property):
        """
        Boilerplate code to create a render buffer producing only a depth texture

        Returns: FrameBuffer for rendering into

        """
        self.depth_tex = Texture("DepthCameraTexture")
        self.depth_tex.setFormat(Texture.FDepthComponent)

        window_props = WindowProperties.size(width, height)
        buffer_props = FrameBufferProperties()

        buffer_props.set_rgba_bits(0, 0, 0, 0)
        buffer_props.set_accum_bits(0)
        buffer_props.set_stencil_bits(0)
        buffer_props.set_back_buffers(0)
        buffer_props.set_coverage_samples(0)
        buffer_props.set_depth_bits(16)
        # buffer_props.set_float_depth(True)

        buffer_props.set_force_hardware(True)
        buffer_props.set_multisamples(0)
        buffer_props.set_srgb_color(False)
        buffer_props.set_stereo(False)
        buffer_props.set_stencil_bits(0)

        buffer = self.engine.graphics_engine.make_output(
            self.engine.win.get_pipe(), self.__class__.__name__, 1, buffer_props, window_props,
            GraphicsPipe.BF_refuse_window, self.engine.win.gsg, self.engine.win
        )

        if buffer is None:
            print("Failed to create buffer")
            return

        buffer.add_render_texture(self.depth_tex, GraphicsOutput.RTM_bind_or_copy, GraphicsOutput.RTP_depth)
        buffer.set_sort(-1000)
        buffer.disable_clears()
        buffer.get_display_region(0).disable_clears()
        buffer.get_overlay_display_region().disable_clears()
        buffer.get_overlay_display_region().set_active(False)

        # Remove all unused display regions
        buffer.remove_all_display_regions()
        buffer.get_display_region(0).set_active(False)
        buffer.disable_clears()

        # Set a clear on the buffer instead on all regions
        buffer.set_clear_depth(1)
        buffer.set_clear_depth_active(True)
        self.buffer = buffer

        # make it for cuda
        self.output_tex = Texture()
        self.output_tex.setup_2d_texture(
            self.depth_tex.getXSize(), self.depth_tex.getYSize(), Texture.T_unsigned_byte, Texture.F_rgba8
        )

        if self.enable_cuda:
            self.output_tex.set_clear_color((0, 0, 0, 1))
            shader = Shader.load_compute(Shader.SL_GLSL, AssetLoader.file_path("../shaders", "depth_convert.glsl"))
            self.compute_node = NodePath("dummy")
            self.compute_node.set_shader(shader)
            # set shader input
            size = LVector2(self.depth_tex.getXSize(), self.depth_tex.getYSize())
            self.compute_node.set_shader_input("fromTex", self.depth_tex)
            self.compute_node.set_shader_input("toTex", self.output_tex)
            self.compute_node.set_shader_input("texSize", size)

    def _dispatch_compute(self, task):
        """
        Call me per frame when you want to access the depth texture result with cuda enabled
        """
        self.engine.graphicsEngine.dispatch_compute(
            (64, 64, 1), self.compute_node.get_attrib(ShaderAttrib), self.engine.win.get_gsg()
        )
        # self.engine.graphicsEngine.extractTextureData(self.output_tex, self.engine.win.get_gsg())
        # self.output_tex.write("{}.png".format(self.engine.episode_step))
        return task.cont

    def get_rgb_array_cpu(self):
        origin_img = self.depth_tex
        self.engine.graphicsEngine.extractTextureData(self.depth_tex, self.engine.win.get_gsg())
        img = np.frombuffer(origin_img.getRamImage().getData(), dtype=np.uint16)
        img = img.reshape((origin_img.getYSize(), origin_img.getXSize(), -1))
        assert img.shape[-1] == 1
        img = img[::-1] / 65535  # to 0-1 float
        img = img[..., :self.num_channels]

        # post proces non-linear depth value
        img = img * 2.0 - 1.0
        img = (2.0 * self.far_near_mul) / (self.far_near_add - img * self.far_near_minus)
        img = np.log(img) / self.log_b - self.log_base_div_b
        return img * 256  # to 0-255

    def add_display_region(self, display_region):
        return
        # if self.engine.mode != "none" and self.display_region is None:
        #     # only show them when onscreen
        #     self.display_region = self.engine.win.makeDisplayRegion(*display_region)
        #     self.display_region.setCamera(self.buffer.getDisplayRegions()[1].camera)
        #     self.draw_border(display_region)

    def _make_cuda_texture(self):
        self.cuda_texture = self.output_tex
