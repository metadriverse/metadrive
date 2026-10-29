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

from metadrive.render_pipeline.rpcore.pluginbase.base_plugin import BasePlugin

from .forward_stage import ForwardStage


class Plugin(BasePlugin):

    name = "Forward Rendering"
    author = "tobspr <tobias.springer1@gmail.com>"
    description = (
        "This plugin adds support for an additional forward rendering "
        "pass. This is mainly useful for transparency."
    )
    version = "0.1 alpha (!)"

    def on_stage_setup(self):
        self.stage = self.create_stage(ForwardStage)

        if self.is_plugin_enabled("scattering"):
            self.stage.required_pipes += ["ScatteringIBLSpecular", "ScatteringIBLDiffuse"]

        if self.is_plugin_enabled("pssm"):
            self.stage.required_pipes += ["PSSMSceneSunShadowMapPCF"]
            self.stage.required_inputs += ["PSSMSceneSunShadowMVP"]

        if self.is_plugin_enabled("env_probes"):
            self.stage.required_pipes += ["PerCellProbes"]
            self.stage.required_inputs += ["EnvProbes"]

    def on_pipeline_created(self):
        pass
