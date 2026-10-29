from panda3d._rplight import PSSMCameraRig
from metadrive.constants import CamMask
from panda3d.core import PTA_LMatrix4
from panda3d.core import Texture, SamplerState
from panda3d.core import WindowProperties, FrameBufferProperties, GraphicsPipe, GraphicsOutput


class PSSM:
    """
    This is the implementation of PSSM for adding shadow for the scene.
    It is based on https://github.com/el-dee/panda3d-samples
    """
    def __init__(self, engine):
        assert engine.world_light, "world_light should be created before having this shadow"

        # engine
        self.engine = engine

        self.camera_rig = None
        self.split_regions = []

        # Basic PSSM configuration
        self.num_splits = 2
        self.split_resolution = 1024
        self.border_bias = 0.058
        self.use_pssm = True
        self.freeze_pssm = False
        self.fog = True
        self.last_cache_reset = engine.clock.get_frame_time()
        self.depth_tex = None
        self.buffer = None

    def init(self):
        """
        Create the PSSM. Lazy init
        Returns: None

        """
        self.create_pssm_camera_rig()
        self.create_pssm_buffer()
        self.attach_pssm_camera_rig()
        self.set_shader_inputs(self.engine.render)
        self.engine.task_mgr.add(self.update)

    @property
    def directional_light(self):
        """
        Return existing directional light
        Returns: Directional Light node path

        """
        return self.engine.world_light.direction_np

    def toggle_shadows_mode(self):
        """
        Switch between shadow casting or not
        Returns: None

        """
        self.use_pssm = not self.use_pssm
        self.engine.render.set_shader_inputs(use_pssm=self.use_pssm)

    def toggle_freeze_pssm(self):
        """
        Stop update shadow
        Returns: None

        """
        self.freeze_pssm = not self.freeze_pssm

    def toggle_fog(self):
        """
        Enable fog
        Returns: None

        """
        self.fog = not self.fog
        self.engine.render.set_shader_inputs(fog=self.fog)

    def update(self, task):
        """
        Engine task for updating shadow caster
        Args:
            task: Panda task, will be filled automatically

        Returns: task.con (task.continue)

        """
        light_dir = self.directional_light.get_mat().xform(-self.directional_light.node().get_direction()).xyz
        self.camera_rig.update(self.engine.camera, light_dir)

        src_mvp_array = self.camera_rig.get_mvp_array()
        mvp_array = PTA_LMatrix4()
        for array in src_mvp_array:
            mvp_array.push_back(array)
        self.engine.render.set_shader_inputs(pssm_mvps=mvp_array)

        # cache_diff = self.engine.clock.get_frame_time() - self.last_cache_reset
        # if cache_diff > 5.0:
        # self.last_cache_reset = self.engine.clock.get_frame_time()
        self.camera_rig.reset_film_size_cache()
        return task.cont

    def create_pssm_camera_rig(self):
        """
        Construct the actual PSSM rig
        Returns: None

        """
        self.camera_rig = PSSMCameraRig(self.num_splits)
        # Set the max distance from the camera where shadows are rendered
        self.camera_rig.set_pssm_distance(self.engine.global_config["shadow_range"])
        # Set the distance between the far plane of the frustum and the sun, objects farther do not cas shadows
        self.camera_rig.set_sun_distance(64)
        # Set the logarithmic factor that defines the splits
        self.camera_rig.set_logarithmic_factor(0.2)

        self.camera_rig.set_border_bias(self.border_bias)
        # Enable CSM splits snapping to avoid shadows flickering when moving
        self.camera_rig.set_use_stable_csm(True)
        # Keep the film size roughly constant to avoid flickering when moving
        self.camera_rig.set_use_fixed_film_size(True)
        # Set the resolution of each split shadow map
        self.camera_rig.set_resolution(self.split_resolution)
        self.camera_rig.reparent_to(self.engine.render)

    def create_pssm_buffer(self):
        """
        Create the depth buffer
        The depth buffer is the concatenation of num_splits shadow maps
        Returns: NOne

        """
        self.depth_tex = Texture("PSSMShadowMap")
        self.depth_tex.setFormat(Texture.FDepthComponent)
        self.depth_tex.setMinfilter(SamplerState.FTShadow)
        self.depth_tex.setMagfilter(SamplerState.FTShadow)
        self.buffer = self.create_render_buffer(self.split_resolution * self.num_splits, self.split_resolution, 32)

        # Remove all unused display regions
        self.buffer.remove_all_display_regions()
        self.buffer.get_display_region(0).set_active(False)
        self.buffer.disable_clears()

        # Set a clear on the buffer instead on all regions
        self.buffer.set_clear_depth(1)
        self.buffer.set_clear_depth_active(True)

        # Prepare the display regions, one for each split
        for i in range(self.num_splits):
            region = self.buffer.make_display_region(
                i / self.num_splits, i / self.num_splits + 1 / self.num_splits, 0, 1
            )
            region.set_sort(25 + i)
            # Clears are done on the buffer
            region.disable_clears()
            region.set_active(True)
            self.split_regions.append(region)

    def attach_pssm_camera_rig(self):
        """
        Attach the cameras to the shadow stage
        Returns: None

        """
        for i in range(self.num_splits):
            camera_np = self.camera_rig.get_camera(i)
            camera_np.node().set_scene(self.engine.render)
            camera_np.node().setCameraMask(CamMask.Shadow)
            self.split_regions[i].set_camera(camera_np)

    def set_shader_inputs(self, target):
        """
        Configure the parameters for the PSSM Shader
        Args:
            target: Target node path to set shader input

        Returns: None

        """
        target.set_shader_inputs(
            PSSMShadowAtlas=self.depth_tex,
            pssm_mvps=self.camera_rig.get_mvp_array(),
            pssm_nearfar=self.camera_rig.get_nearfar_array(),
            border_bias=self.border_bias,
            use_pssm=self.use_pssm,
            fog=self.fog,
            split_count=self.num_splits,
            light_direction=self.engine.world_light.direction_pos
        )

    def create_render_buffer(self, size_x, size_y, depth_bits):
        """
        Boilerplate code to create a render buffer producing only a depth texture
        Args:
            size_x: Render buffer size x
            size_y: Render buffer size y
            depth_bits: bit for Depth test
            depth_tex: Deprecated

        Returns: FrameBuffer for rendering into

        """
        window_props = WindowProperties.size(size_x, size_y)
        buffer_props = FrameBufferProperties()

        buffer_props.set_rgba_bits(0, 0, 0, 0)
        buffer_props.set_accum_bits(0)
        buffer_props.set_stencil_bits(0)
        buffer_props.set_back_buffers(0)
        buffer_props.set_coverage_samples(0)
        buffer_props.set_depth_bits(depth_bits)

        if depth_bits == 32:
            buffer_props.set_float_depth(True)

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

        buffer.set_sort(-1001)
        buffer.disable_clears()
        buffer.get_display_region(0).disable_clears()
        buffer.get_overlay_display_region().disable_clears()
        buffer.get_overlay_display_region().set_active(False)

        return buffer
