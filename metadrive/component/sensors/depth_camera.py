from panda3d.core import Texture, SamplerState
from panda3d.core import WindowProperties, FrameBufferProperties, GraphicsPipe, GraphicsOutput

from metadrive.component.sensors.base_camera import BaseCamera
from metadrive.constants import CamMask


class DepthCamera(BaseCamera):
    CAM_MASK = CamMask.DepthCam

    GROUND_HEIGHT = 0
    VIEW_GROUND = True
    GROUND = None
    GROUND_MODEL = None

    num_channels = 1
    shader_name = "depth_cam"

    def __init__(self, width, height, engine, *, cuda=False):
        self.BUFFER_W, self.BUFFER_H = width, height
        super(DepthCamera, self).__init__(engine, cuda)

        # post set camera
        self.cam.node().set_scene(self.engine.render)
        self.buffer_display_region.set_camera(self.cam)

        cam = self.get_cam()
        lens = self.get_lens()
        lens.setFov(60)

        # cam.lookAt(0, 2.4, 1.3)
        cam.lookAt(0, 10.4, 1.6)

    def _create_buffer(self, width, height, frame_buffer_property):
        """
        Boilerplate code to create a render buffer producing only a depth texture

        Returns: FrameBuffer for rendering into

        """
        self.depth_tex = Texture("PSSMShadowMap")
        self.depth_tex.setFormat(Texture.FDepthComponent)
        self.depth_tex.setMinfilter(SamplerState.FTShadow)
        self.depth_tex.setMagfilter(SamplerState.FTShadow)

        window_props = WindowProperties.size(width, height)
        buffer_props = FrameBufferProperties()

        buffer_props.set_rgba_bits(0, 0, 0, 0)
        buffer_props.set_accum_bits(0)
        buffer_props.set_stencil_bits(0)
        buffer_props.set_back_buffers(0)
        buffer_props.set_coverage_samples(0)
        buffer_props.set_depth_bits(8)

        # if depth_bits == 32:
        # buffer_props.set_float_depth(True)

        buffer_props.set_force_hardware(True)
        buffer_props.set_multisamples(0)
        buffer_props.set_srgb_color(False)
        buffer_props.set_stereo(False)
        buffer_props.set_stencil_bits(0)

        buffer = self.engine.graphics_engine.make_output(
            self.engine.win.get_pipe(), "pssm_buffer", 1, buffer_props, window_props, GraphicsPipe.BF_refuse_window,
            self.engine.win.gsg, self.engine.win
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

        # Prepare the display regions, one for each split
        region = buffer.make_display_region(0, 1, 0, 1)
        region.set_sort(25)
        # Clears are done on the buffer
        region.disable_clears()
        region.set_active(True)
        self.buffer_display_region = region
        return buffer
