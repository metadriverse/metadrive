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

from __future__ import division

from metadrive.render_pipeline.rplibs.six import itervalues

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.render_stage import RenderStage


class SharpenStage(RenderStage):

    required_inputs = []
    required_pipes = ["ShadedScene"]

    def __init__(self, pipeline):
        RenderStage.__init__(self, pipeline)
        self.sharpen_twice = True

    @property
    def produced_pipes(self):
        if self.sharpen_twice:
            return {"ShadedScene": self.target2.color_tex}
        else:
            return {"ShadedScene": self.target.color_tex}

    def create(self):
        native_size = Globals.native_resolution.x, Globals.native_resolution.y
        self.target = self.create_target("Sharpen")
        self.target.size = native_size
        self.target.add_color_attachment(bits=16)
        self.target.prepare_buffer()

        if self.sharpen_twice:
            self.target2 = self.create_target("Sharpen2")
            self.target2.size = native_size
            self.target2.add_color_attachment(bits=16)
            self.target2.prepare_buffer()
            self.target2.set_shader_input("ShadedScene", self.target.color_tex)

    def set_dimensions(self):
        self.target.size = Globals.native_resolution.x, Globals.native_resolution.y
        if self.sharpen_twice:
            self.target2.size = Globals.native_resolution.x, Globals.native_resolution.y

    def reload_shaders(self):
        sharpen_shader = self.load_plugin_shader("sharpen.frag.glsl")
        for target in itervalues(self._targets):
            target.shader = sharpen_shader
