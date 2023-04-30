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
from metadrive.render_pipeline.rpcore.stages.ambient_stage import AmbientStage

from .capture_stage import SkyAOCaptureStage
from .ao_stage import SkyAOStage


class Plugin(BasePlugin):

    name = "Sky Occlusion"
    author = "tobspr <tobias.springer1@gmail.com>"
    description = ("This plugin adds support for sky occlusion, computing " "occlusion based on the sky visibility.")
    version = "0.2 beta"

    def on_stage_setup(self):
        self.capture_stage = self.create_stage(SkyAOCaptureStage)
        self.ao_stage = self.create_stage(SkyAOStage)

        self.capture_stage.resolution = self.get_setting("resolution")
        self.capture_stage.max_radius = self.get_setting("max_radius")
        self.capture_stage.capture_height = self.get_setting("capture_height")

        # Make the stages use our output
        AmbientStage.required_pipes.append("SkyAO")

    def on_post_stage_setup(self):
        if self.is_plugin_enabled("env_probes"):
            self.get_plugin_instance("env_probes").capture_stage.required_inputs.append("SkyAOCapturePosition")
            self.get_plugin_instance("env_probes").capture_stage.required_pipes.append("SkyAOHeight")

    def on_pipeline_created(self):
        pass
