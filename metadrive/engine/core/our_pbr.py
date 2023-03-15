# in order to use pbr in opengles pipe on clusters, we temporally inherit from simple pbr
import panda3d.core as p3d

from panda3d.core import Shader, ConfigVariableString
from simplepbr import Pipeline, _add_shader_defines, _load_shader_str

from metadrive.engine.asset_loader import AssetLoader

# def _load_shader_str(shaderpath, defines=None):
#     shader_dir = AssetLoader.file_path("shaders", "pbr_shaders", shaderpath)
#
#     with open(shader_dir) as shaderfile:
#         shaderstr = shaderfile.read()
#
#     if defines is not None:
#         shaderstr = _add_shader_defines(shaderstr, defines)
#
#     return shaderstr


class OurPipeline(Pipeline):
    # raise DeprecationWarning("This feature is deprecated now")

    def __init__(
        self,
        render_node=None,
        window=None,
        camera_node=None,
        taskmgr=None,
        msaa_samples=4,
        max_lights=8,
        use_normal_maps=False,
        use_emission_maps=True,
        exposure=1.0,
        enable_shadows=False,
        enable_fog=False,
        use_occlusion_maps=False
    ):
        super(OurPipeline, self).__init__(
            render_node=render_node,
            window=window,
            camera_node=camera_node,
            taskmgr=taskmgr,
            msaa_samples=msaa_samples,
            max_lights=max_lights,
            use_normal_maps=use_normal_maps,
            use_emission_maps=use_emission_maps,
            exposure=exposure,
            enable_shadows=enable_shadows,
            enable_fog=enable_fog,
            use_occlusion_maps=use_occlusion_maps
        )

    def _setup_tonemapping(self):
        # return
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
        fbprops.set_rgba_bits(16, 16, 16, 16)
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

    def _recompile_pbr(self):
        gles = ConfigVariableString("load-display").getValue()
        if gles == "pandagles2":
            pbr_defines = {
                'MAX_LIGHTS': self.max_lights,
            }
            if self.use_normal_maps:
                pbr_defines['USE_NORMAL_MAP'] = ''
            if self.use_emission_maps:
                pbr_defines['USE_EMISSION_MAP'] = ''
            if self.enable_shadows:
                pbr_defines['ENABLE_SHADOWS'] = ''
            if self.enable_fog:
                pbr_defines['ENABLE_FOG'] = ''
            if self.use_occlusion_maps:
                pbr_defines['USE_OCCLUSION_MAP'] = ''

            pbr_vert_str = _load_shader_str('simplepbr_gles.vert', pbr_defines)
            pbr_frag_str = _load_shader_str('simplepbr_gles.frag', pbr_defines)
            pbrshader = Shader.make(
                Shader.SL_GLSL,
                vertex=pbr_vert_str,
                fragment=pbr_frag_str,
            )
            self.render_node.set_shader(pbrshader)
        else:
            super(OurPipeline, self)._recompile_pbr()
