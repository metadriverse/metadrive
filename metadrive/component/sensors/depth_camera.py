import numpy as np
from panda3d.core import Camera
from panda3d.core import CardMaker
from panda3d.core import OrthographicLens
from panda3d.core import Texture, NodePath
from panda3d.core import WindowProperties, FrameBufferProperties, GraphicsPipe, GraphicsOutput

from metadrive.component.sensors.base_camera import BaseCamera
from metadrive.constants import CamMask
from metadrive.constants import Semantics, CameraTagStateKey


class DepthCamera(BaseCamera):
    CAM_MASK = CamMask.DepthCam

    num_channels = 1

    def __init__(self, width, height, engine, *, cuda=False):
        self.BUFFER_W, self.BUFFER_H = width, height

        # create
        super(DepthCamera, self).__init__(engine, cuda)
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

        self.buffer = self.engine.graphicsEngine.makeOutput(
            self.engine.pipe, "Depth buffer", -2,
            props, window_props,
            GraphicsPipe.BFRefuseWindow,
            self.engine.win.getGsg(), self.engine.win)
        mode =  GraphicsOutput.RTMBindOrCopy if self._enable_cuda else GraphicsOutput.RTMCopyRam
        self.buffer.addRenderTexture(self.depth_tex, mode,
                                     GraphicsOutput.RTPDepthStencil)

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

    @staticmethod
    def _format(ret, to_float):
        if not to_float:
            ret = (ret * 255).astype(np.uint8)
        return ret

    def _make_cuda_texture(self):
        """
        Decide which texture to retrieve on GPU
        """
        self.cuda_texture = self.depth_tex
