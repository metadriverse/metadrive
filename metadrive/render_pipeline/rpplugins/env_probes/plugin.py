"""

RenderPipeline

Copyright (c) 2014-2016 tobspr <tobias.springer1@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

"""

from panda3d.core import PTAInt

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.util.shader_input_blocks import SimpleInputBlock
from metadrive.render_pipeline.rpcore.pluginbase.base_plugin import BasePlugin
from metadrive.render_pipeline.rpcore.stages.cull_lights_stage import CullLightsStage

from .probe_manager import ProbeManager
from .environment_capture_stage import EnvironmentCaptureStage
from .apply_envprobes_stage import ApplyEnvprobesStage
from .cull_probes_stage import CullProbesStage


class Plugin(BasePlugin):

    name = "Environment Probes"
    author = "tobspr <tobias.springer1@gmail.com>"
    description = (
        "This plugin adds support for environment probes, containing "
        "diffuse and specular information. This enables accurate "
        "reflections, and can also be used to simulate GI."
    )
    version = "beta (!)"

    def on_stage_setup(self):
        self.probe_mgr = ProbeManager()
        self.probe_mgr.resolution = self.get_setting("probe_resolution")
        self.probe_mgr.diffuse_resolution = self.get_setting("diffuse_probe_resolution")
        self.probe_mgr.max_probes = self.get_setting("max_probes")
        self.probe_mgr.init()
        self._setup_stages()

    def _setup_stages(self):
        """ Setups all stages """
        # Create the stage to generate and update the cubemaps
        self.capture_stage = self.create_stage(EnvironmentCaptureStage)
        self.capture_stage.resolution = self.probe_mgr.resolution
        self.capture_stage.diffuse_resolution = self.probe_mgr.diffuse_resolution
        self.capture_stage.storage_tex = self.probe_mgr.cubemap_storage
        self.capture_stage.storage_tex_diffuse = self.probe_mgr.diffuse_storage

        # Create the stage to cull the cubemaps
        self.cull_stage = self.create_stage(CullProbesStage)

        # Create the stage to apply the cubemaps
        self.apply_stage = self.create_stage(ApplyEnvprobesStage)

        if self.is_plugin_enabled("scattering"):
            self.capture_stage.required_pipes += ["ScatteringIBLSpecular", "ScatteringIBLDiffuse"]

        if self.is_plugin_enabled("pssm"):
            self.capture_stage.required_pipes += ["PSSMSceneSunShadowMapPCF"]
            self.capture_stage.required_inputs += ["PSSMSceneSunShadowMVP"]

        self._setup_inputs()

    def _setup_inputs(self):
        """ Sets all required inputs """
        self.pta_probes = PTAInt.empty_array(1)

        # Construct the UBO which stores all environment probe data
        self.data_ubo = SimpleInputBlock("EnvProbes")
        self.data_ubo.add_input("num_probes", self.pta_probes)
        self.data_ubo.add_input("cubemaps", self.probe_mgr.cubemap_storage)
        self.data_ubo.add_input("diffuse_cubemaps", self.probe_mgr.diffuse_storage)
        self.data_ubo.add_input("dataset", self.probe_mgr.dataset_storage)
        self._pipeline.stage_mgr.input_blocks.append(self.data_ubo)

        # Use the UBO in light culling
        CullLightsStage.required_inputs.append("EnvProbes")

    def on_pre_render_update(self):
        if self._pipeline.task_scheduler.is_scheduled("envprobes_select_and_cull"):
            self.probe_mgr.update()
            self.pta_probes[0] = self.probe_mgr.num_probes
            probe = self.probe_mgr.find_probe_to_update()
            if probe:
                probe.last_update = Globals.clock.get_frame_count()
                self.capture_stage.active = True
                self.capture_stage.set_probe(probe)

                if self.is_plugin_enabled("pssm"):
                    self.get_plugin_instance("pssm").scene_shadow_stage.request_focus(
                        probe.bounds.get_center(), probe.bounds.get_radius()
                    )
            else:
                self.capture_stage.active = False
