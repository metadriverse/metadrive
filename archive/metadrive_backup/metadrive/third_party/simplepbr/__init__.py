import os

import panda3d.core as p3d

from direct.filter.FilterManager import FilterManager

from .version import __version__

# try:
#     from .shaders import shaders
# except ImportError:
shaders = None

__all__ = ['init', 'Pipeline']


def _add_shader_defines(shaderstr, defines):
    """
    Add Define for enabling some functions
    Args:
        shaderstr:
        defines:

    Returns:

    """
    shaderlines = shaderstr.split('\n')

    for line in shaderlines:
        if '#version' in line:
            version_line = line
            break
    else:
        raise RuntimeError('Failed to find GLSL version string')
    shaderlines.remove(version_line)

    define_lines = [f'#define {define} {value}' for define, value in defines.items()]

    return '\n'.join([version_line] + define_lines + ['#line 1'] + shaderlines)


def _load_shader_str(shaderpath, defines=None):
    """
    Load the shader as string from the shaders dir (instead of shaders.py)
    Args:
        shaderpath:
        defines:

    Returns:

    """
    if shaders:
        shaderstr = shaders[shaderpath]
    else:
        shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')

        with open(os.path.join(shader_dir, shaderpath + ".glsl")) as shaderfile:
            shaderstr = shaderfile.read()

    if defines is None:
        defines = {}

    defines['p3d_TextureBaseColor'] = 'p3d_TextureModulate'
    defines['p3d_TextureMetalRoughness'] = 'p3d_TextureSelector'
    defines['p3d_TextureNormal'] = 'p3d_TextureNormal'
    defines['p3d_TextureEmission'] = 'p3d_TextureEmission'

    shaderstr = _add_shader_defines(shaderstr, defines)

    if 'USE_330' in defines:
        shaderstr = shaderstr.replace('#version 120', '#version 330')
        if shaderpath.endswith('vert'):
            shaderstr = shaderstr.replace('varying ', 'out ')
            shaderstr = shaderstr.replace('attribute ', 'in ')
        else:
            shaderstr = shaderstr.replace('varying ', 'in ')

    return shaderstr


class Pipeline:
    """
    A SimplePBR Pipeline
    """
    def __init__(
        self,
        *,
        render_node=None,
        window=None,
        camera_node=None,
        taskmgr=None,
        msaa_samples=4,
        max_lights=8,
        use_normal_maps=False,
        use_emission_maps=True,
        exposure=1.0,
        enable_fog=False,
        use_occlusion_maps=False,
        use_330=None,
        use_hardware_skinning=None,
    ):
        if render_node is None:
            render_node = base.render

        if window is None:
            window = base.win

        if camera_node is None:
            camera_node = base.cam

        if taskmgr is None:
            taskmgr = base.task_mgr

        self._shader_ready = False
        self.render_node = render_node
        self.window = window
        self.camera_node = camera_node
        self.max_lights = max_lights
        self.use_normal_maps = use_normal_maps
        self.use_emission_maps = use_emission_maps
        self.enable_fog = enable_fog
        self.exposure = exposure
        self.msaa_samples = msaa_samples
        self.use_occlusion_maps = use_occlusion_maps

        self._set_use_330(use_330)
        self.enable_hardware_skinning = use_hardware_skinning if use_hardware_skinning is not None else self.use_330

        # Create a FilterManager instance
        self.manager = FilterManager(window, camera_node)

        # Do not force power-of-two textures
        p3d.Texture.set_textures_power_2(p3d.ATS_none)

        # Make sure we have AA for if/when MSAA is enabled
        self.render_node.set_antialias(p3d.AntialiasAttrib.M_auto)

        # PBR Shader
        self._recompile_pbr()

        # Tonemapping
        self._setup_tonemapping()

        self._shader_ready = True

    def _set_use_330(self, use_330):
        """
        Use shader version 330. We already enable it in shader. So don't call this API
        Args:
            use_330:

        Returns:

        """
        if use_330 is not None:
            self.use_330 = use_330
        else:
            self.use_330 = False

            cvar = p3d.ConfigVariableInt('gl-version')
            gl_version = [cvar.get_word(i) for i in range(cvar.get_num_words())]
            if len(gl_version) >= 2 and gl_version[0] >= 3 and gl_version[1] >= 2:
                # Not exactly accurate, but setting this variable to '3 2' is common for disabling
                # the fixed-function pipeline and 3.2 support likely means 3.3 support as well.
                self.use_330 = True

    def __setattr__(self, name, value):
        """
        Reload shader if required
        Args:
            name:
            value:

        Returns:

        """
        if hasattr(self, name):
            prev_value = getattr(self, name)
        else:
            prev_value = None
        super().__setattr__(name, value)
        if not self._shader_ready:
            return

        pbr_vars = [
            'max_lights',
            'use_normal_maps',
            'use_emission_maps',
            'enable_fog',
            'use_occlusion_maps',
        ]

        def resetup_tonemap():
            # Destroy previous buffers so we can re-create
            self.manager.cleanup()

            # Create a new FilterManager instance
            self.manager = FilterManager(self.window, self.camera_node)
            self._setup_tonemapping()

        if name in pbr_vars and prev_value != value:
            self._recompile_pbr()
        elif name == 'exposure':
            self.tonemap_quad.set_shader_input('exposure', self.exposure)
        elif name == 'msaa_samples':
            self._setup_tonemapping()
        elif name == 'render_node' and prev_value != value:
            self._recompile_pbr()
        elif name in ('camera_node', 'window') and prev_value != value:
            resetup_tonemap()
        elif name == 'use_330' and prev_value != value:
            self._set_use_330(value)
            self._recompile_pbr()
            resetup_tonemap()

    def _recompile_pbr(self):
        """
        Recompile and reload PBR
        Returns:

        """
        pbr_defines = {
            'MAX_LIGHTS': self.max_lights,
        }
        if self.use_normal_maps:
            pbr_defines['USE_NORMAL_MAP'] = ''
        if self.use_emission_maps:
            pbr_defines['USE_EMISSION_MAP'] = ''
        if self.enable_fog:
            pbr_defines['ENABLE_FOG'] = ''
        if self.use_occlusion_maps:
            pbr_defines['USE_OCCLUSION_MAP'] = ''
        if self.use_330:
            pbr_defines['USE_330'] = ''
        if self.enable_hardware_skinning:
            pbr_defines['ENABLE_SKINNING'] = ''

        pbr_vert_str = _load_shader_str('simplepbr.vert', pbr_defines)
        pbr_frag_str = _load_shader_str('simplepbr.frag', pbr_defines)
        pbrshader = p3d.Shader.make(
            p3d.Shader.SL_GLSL,
            vertex=pbr_vert_str,
            fragment=pbr_frag_str,
        )
        attr = p3d.ShaderAttrib.make(pbrshader)
        if self.enable_hardware_skinning:
            attr = attr.set_flag(p3d.ShaderAttrib.F_hardware_skinning, True)
        self.render_node.set_attrib(attr)

    def _setup_tonemapping(self):
        """
        Use tonemapping to correct the color
        Returns:

        """
        if self._shader_ready:
            # Destroy previous buffers so we can re-create
            self.manager.cleanup()

            # Fix shadow buffers after FilterManager.cleanup()
            for caster in self.get_all_casters():
                sbuff_size = caster.get_shadow_buffer_size()
                caster.set_shadow_buffer_size((0, 0))
                caster.set_shadow_buffer_size(sbuff_size)

        fbprops = p3d.FrameBufferProperties()
        fbprops.float_color = True
        # fbprops.set_rgba_bits(16, 16, 16, 16)
        fbprops.set_depth_bits(24)
        fbprops.set_multisamples(self.msaa_samples)
        scene_tex = p3d.Texture()
        scene_tex.set_format(p3d.Texture.F_rgba16)
        scene_tex.set_component_type(p3d.Texture.T_float)
        self.tonemap_quad = self.manager.render_scene_into(colortex=scene_tex, fbprops=fbprops)

        defines = {}
        if self.use_330:
            defines['USE_330'] = ''

        post_vert_str = _load_shader_str('post.vert', defines)
        post_frag_str = _load_shader_str('tonemap.frag', defines)
        tonemap_shader = p3d.Shader.make(
            p3d.Shader.SL_GLSL,
            vertex=post_vert_str,
            fragment=post_frag_str,
        )
        self.tonemap_quad.set_shader(tonemap_shader)
        self.tonemap_quad.set_shader_input('tex', scene_tex)
        self.tonemap_quad.set_shader_input('exposure', self.exposure)

    def get_all_casters(self):
        """
        Get shader caster
        Returns:

        """
        engine = p3d.GraphicsEngine.get_global_ptr()
        cameras = [dispregion.camera for win in engine.windows for dispregion in win.active_display_regions]

        return [i.node() for i in cameras if hasattr(i.node(), 'is_shadow_caster') and i.node().is_shadow_caster()]

    def verify_shaders(self):
        """
        Verify shader
        Returns:

        """
        gsg = self.window.gsg

        def check_node_shader(np):
            shader = p3d.Shader(np.get_shader())
            shader.prepare_now(gsg.prepared_objects, gsg)
            assert shader.is_prepared(gsg.prepared_objects)
            assert not shader.get_error_flag()

        check_node_shader(self.render_node)
        check_node_shader(self.tonemap_quad)


def init(**kwargs):
    '''Initialize the PBR render pipeline
    :param render_node: The node to attach the shader too, defaults to `base.render` if `None`
    :type render_node: `panda3d.core.NodePath`
    :param window: The window to attach the framebuffer too, defaults to `base.win` if `None`
    :type window: `panda3d.core.GraphicsOutput
    :param camera_node: The NodePath of the camera to use when rendering the scene, defaults to `base.cam` if `None`
    :type camera_node: `panda3d.core.NodePath
    :param msaa_samples: The number of samples to use for multisample anti-aliasing, defaults to 4
    :type msaa_samples: int
    :param max_lights: The maximum number of lights to render, defaults to 8
    :type max_lights: int
    :param use_normal_maps: Use normal maps, defaults to `False` (NOTE: Requires models with appropriate tangents)
    :type use_normal_maps: bool
    :param use_emission_maps: Use emission maps, defaults to `True`
    :type use_emission_maps: bool
    :param exposure: a value used to multiply the screen-space color value prior to tonemapping, defaults to 1.0
    :type exposure: float
    :param enable_fog: Enable exponential fog, defaults to False
    :type enable_fog: bool
    :param use_occlusion_maps: Use occlusion maps, defaults to `False` (NOTE: Requires occlusion channel in
    metal-roughness map)
    :type use_occlusion_maps: bool
    :param use_330: Force the usage of GLSL 330 shaders (version 120 otherwise, auto-detect if None)
    :type use_330: bool or None
    :param use_hardware_skinning: Force usage of hardware skinning for skeleton animations
        (auto-detect if None, defaults to None)
    :type use_hardware_skinning: bool or None
    '''

    return Pipeline(**kwargs)
