import numpy as np
from panda3d.core import Camera
from panda3d.core import CardMaker
from panda3d.core import OrthographicLens
from panda3d.core import Texture, Shader, NodePath, ShaderAttrib, LVector2
from panda3d.core import WindowProperties, FrameBufferProperties, GraphicsPipe, GraphicsOutput

from metadrive.component.sensors.base_camera import BaseCamera
from metadrive.constants import CamMask
from metadrive.constants import Semantics, CameraTagStateKey
from metadrive.engine.asset_loader import AssetLoader


class DepthCamera(BaseCamera):
    CAM_MASK = CamMask.DepthCam

    num_channels = 1

    def __init__(self, width, height, engine, *, cuda=False):
        """
        The depth camera needs OpenGL > 4.3 if you want to use CUDA
        """
        self.BUFFER_W, self.BUFFER_H = width, height
        # create
        super(DepthCamera, self).__init__(engine, cuda)
        if self.enable_cuda:
            self.shader_local_size = (16, 16)
            self.engine.taskMgr.add(self._dispatch_compute)

        # display region
        self.quad = None
        self.quadcam = None

    def _setup_effect(self):
        """
        Set compute shader input
        """
        cam = self.get_cam().node()
        cam.setTagStateKey(CameraTagStateKey.Depth)
        from metadrive.engine.core.terrain import Terrain
        cam.setTagState(
            Semantics.TERRAIN.label,
            Terrain.make_render_state(self.engine, "terrain.vert.glsl", "terrain_depth.frag.glsl")
        )
        if self.enable_cuda:
            size = LVector2(self.depth_tex.getXSize(), self.depth_tex.getYSize())
            self.compute_node.set_shader_input("fromTex", self.depth_tex)
            self.compute_node.set_shader_input("toTex", self.output_tex)
            self.compute_node.set_shader_input("texSize", size)

    def _create_camera(self, pos, bkg_color):
        """
        Create camera for the buffer
        """
        super(DepthCamera, self)._create_camera(pos, bkg_color)
        self.cam.node().set_scene(self.engine.render)
        self.buffer_display_region = self.cam.node().getDisplayRegion(0)
        self.buffer_display_region.set_sort(25)
        self.buffer_display_region.disable_clears()
        self.buffer_display_region.set_active(True)

    def _create_buffer(self, width, height, frame_buffer_property):
        """
        Boilerplate code to create a render buffer producing only a depth texture

        Returns: FrameBuffer for rendering into

        """
        self.depth_tex = Texture("DepthCameraTexture")
        self.depth_tex.setFormat(Texture.FDepthComponent)

        window_props = WindowProperties.size(width, height)
        props = FrameBufferProperties()
        props.setRgbColor(0)
        props.setAlphaBits(0)
        props.setDepthBits(1)

        buffer_props = props
        buffer_props.set_rgba_bits(0, 0, 0, 0)
        # buffer_props.set_accum_bits(0)
        # buffer_props.set_stencil_bits(0)
        # buffer_props.set_back_buffers(0)
        # buffer_props.set_coverage_samples(0)
        buffer_props.set_float_depth(True)

        self.buffer = buffer = self.engine.graphicsEngine.makeOutput(
            self.engine.pipe, "Depth buffer", -2, props, window_props, GraphicsPipe.BFRefuseWindow,
            self.engine.win.getGsg(), self.engine.win
        )
        mode = GraphicsOutput.RTMBindOrCopy if self._enable_cuda else GraphicsOutput.RTMCopyRam
        self.buffer.addRenderTexture(self.depth_tex, mode, GraphicsOutput.RTPDepth)
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

        if self.enable_cuda:
            self.output_tex = Texture()
            self.output_tex.setup_2d_texture(
                self.depth_tex.getXSize(), self.depth_tex.getYSize(), Texture.T_unsigned_byte, Texture.F_rgba8
            )
            self.output_tex.set_clear_color((0, 0, 0, 1))
            shader = Shader.load_compute(Shader.SL_GLSL, AssetLoader.file_path("../shaders", "depth_cuda.glsl"))
            self.compute_node = NodePath("dummy")
            self.compute_node.set_shader(shader)

    def _dispatch_compute(self, task):
        """
        Call me per frame when you want to access the depth texture result with cuda enabled
        """
        work_group_x = int(np.ceil(self.depth_tex.getXSize() / self.shader_local_size[0]))
        work_group_y = int(np.ceil(self.depth_tex.getYSize() / self.shader_local_size[1]))
        self.engine.graphicsEngine.dispatch_compute(
            (work_group_x, work_group_y, 1), self.compute_node.get_attrib(ShaderAttrib), self.engine.win.get_gsg()
        )
        # self.engine.graphicsEngine.extractTextureData(self.output_tex, self.engine.win.get_gsg())
        # self.output_tex.write("{}.png".format(self.engine.episode_step))
        return task.cont

    def get_rgb_array_cpu(self):
        """
        Moving the texture to RAM and turn it into numpy array
        Returns:

        """
        origin_img = self.depth_tex
        img = np.frombuffer(origin_img.getRamImage().getData(), dtype=np.float32)
        img = img.reshape((origin_img.getYSize(), origin_img.getXSize(), -1))
        img = img[..., :self.num_channels]
        assert img.shape[-1] == 1
        img = img[::-1]
        return img

    def add_display_region(self, display_region, keep_height=True):
        """
        Add a display region to show the rendering result
        """
        if self.engine.mode != "none" and self.display_region is None:
            if keep_height:
                ratio = self.BUFFER_H / self.BUFFER_W
                h = 0.333 * ratio
                display_region[-2] = 1 - h

            cm = CardMaker("filter-base-quad")
            cm.setFrameFullscreenQuad()
            self.quad = quad = NodePath(cm.generate())
            quad.setDepthTest(0)
            quad.setDepthWrite(0)
            quad.setTexture(self.depth_tex)

            quadcamnode = Camera("depth_result_cam")
            lens = OrthographicLens()
            lens.setFilmSize(2, 2)
            lens.setFilmOffset(0, 0)
            lens.setNearFar(-1000, 1000)
            quadcamnode.setLens(lens)
            self.quadcam = quad.attachNewNode(quadcamnode)

            # only show them when onscreen
            self.display_region = self.engine.win.makeDisplayRegion(*display_region)
            self.display_region.setCamera(self.quadcam)
            self.draw_border(display_region)

    def remove_display_region(self):
        """
        Remove the display region
        """
        if self.quadcam is not None:
            self.quadcam.removeNode()
            self.quad.removeNode()
        super(DepthCamera, self).remove_display_region()

    def _format(self, ret, to_float):
        if self.enable_cuda:
            return super()._format(ret, to_float)

        # else
        if not to_float:
            ret = (ret * 255).astype(np.uint8)
        return ret

    def _make_cuda_texture(self):
        self.cuda_texture = self.output_tex
