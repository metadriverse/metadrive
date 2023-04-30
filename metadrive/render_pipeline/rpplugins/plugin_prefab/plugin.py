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
# Load the base plugin class
from metadrive.render_pipeline.rpcore.pluginbase.base_plugin import BasePlugin

# Load your additional plugin classes here, if required
from .demo_stage import DemoStage


class Plugin(BasePlugin):

    name = "Plugin Prefab"
    author = "tobspr <tobias.springer1@gmail.com>"
    description = ("This is the most basic structure of a plugin. You can copy " "it to produce your own plugins")
    version = "1.0"

    def on_stage_setup(self):
        """ This method gets called when the pipeline setups the render
        stages. You should create your custom stages here """

        # Setup a demo stage
        self.stage = self.create_stage(DemoStage)

    def on_pipeline_created(self):
        """ This method gets called after the pipeline finished the setup,
        and is about to start rendering """

    def update_some_setting(self):
        """ This method gets called when the setting "some_setting"
        of your plugin gets called. You should do all work to update required
        inputs etc. yourself. """
