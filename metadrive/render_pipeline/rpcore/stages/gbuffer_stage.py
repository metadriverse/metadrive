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

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.render_stage import RenderStage
from metadrive.render_pipeline.rpcore.util.shader_input_blocks import SimpleInputBlock


class GBufferStage(RenderStage):
    """ This is the main pass stage, rendering the objects and creating the
    GBuffer which is used in later stages """

    required_inputs = ["DefaultEnvmap"]
    required_pipes = []

    @property
    def produced_pipes(self):
        return {"GBuffer": self.make_gbuffer_ubo(), "SceneDepth": self.target.depth_tex}

    def make_gbuffer_ubo(self):
        ubo = SimpleInputBlock("GBuffer")
        ubo.add_input("Depth", self.target.depth_tex)
        ubo.add_input("Data0", self.target.color_tex)
        ubo.add_input("Data1", self.target.aux_tex[0])
        ubo.add_input("Data2", self.target.aux_tex[1])
        return ubo

    def create(self):
        self.target = self.create_target("GBuffer")
        self.target.add_color_attachment(bits=16, alpha=True)
        self.target.add_depth_attachment(bits=32)
        self.target.add_aux_attachments(bits=16, count=2)
        self.target.prepare_render(Globals.base.cam)

    def set_shader_input(self, *args):
        Globals.render.set_shader_input(*args)

    def set_shader_inputs(self, **kwargs):
        Globals.render.set_shader_inputs(**kwargs)
