# in order to use pbr in opengles pipe on clusters, we temporally inherit from simple pbr

from panda3d.core import Shader, ConfigVariableString
from simplepbr import Pipeline, _add_shader_defines

from metadrive.engine.asset_loader import AssetLoader


def _load_shader_str(shaderpath, defines=None):
    shader_dir = AssetLoader.file_path("shaders", "pbr_shaders", shaderpath)

    with open(shader_dir) as shaderfile:
        shaderstr = shaderfile.read()

    if defines is not None:
        shaderstr = _add_shader_defines(shaderstr, defines)

    return shaderstr


class OurPipeline(Pipeline):
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
        # this func cause error under opengles model
        pass

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
