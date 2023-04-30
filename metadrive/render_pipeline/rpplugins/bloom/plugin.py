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

from metadrive.render_pipeline.rpcore.loader import RPLoader
from metadrive.render_pipeline.rpcore.pluginbase.base_plugin import BasePlugin

from .bloom_stage import BloomStage


class Plugin(BasePlugin):

    name = "Bloom"
    author = "tobspr <tobias.springer1@gmail.com>"
    description = (
        "This plugin adds support for a technique called Bloom, which "
        "makes very bright objects like the sun have a halo."
    )
    version = "1.2"

    def on_stage_setup(self):
        self._bloom_stage = self.create_stage(BloomStage)
        self._bloom_stage.num_mips = self.get_setting("num_mipmaps")
        self._bloom_stage.remove_fireflies = self.get_setting("remove_fireflies")

    def on_pipeline_created(self):
        dirt_tex = RPLoader.load_texture(self.get_resource("lens_dirt.txo"))
        self._bloom_stage.set_shader_input("LensDirtTex", dirt_tex)
