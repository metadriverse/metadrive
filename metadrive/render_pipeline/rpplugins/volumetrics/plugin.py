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
from .volumetrics_stage import VolumetricsStage


class Plugin(BasePlugin):

    name = "Volumetric Lighting"
    author = "tobspr <tobias.springer1@gmail.com>"
    description = ("This plugins adds support for volumetric lighting")
    version = "0.1 alpha (!)"

    required_plugins = ("pssm", )

    def on_stage_setup(self):
        self.stage = self.create_stage(VolumetricsStage)
        self.stage.enable_volumetric_shadows = self.get_setting("enable_volumetric_shadows")

        if self.get_plugin_instance("pssm").get_setting("use_pcf"):
            self.stage.required_pipes.append("PSSMShadowAtlasPCF")
        else:
            self.stage.required_pipes.append("PSSMShadowAtlas")
