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

from panda3d.core import Vec2

from metadrive.render_pipeline.rpcore.render_stage import RenderStage


class SkyAOStage(RenderStage):
    """ This is the main stage used by the Sky AO plugin, and computes
    the sky occlusion term """

    required_inputs = ["SkyAOCapturePosition"]
    required_pipes = ["SkyAOHeight", "GBuffer"]

    @property
    def produced_pipes(self):
        return {"SkyAO": self.target_upscale.color_tex}

    def create(self):
        self.target = self.create_target("ComputeSkyAO")
        self.target.size = -2
        self.target.add_color_attachment(bits=(16, 0, 0, 0))
        self.target.prepare_buffer()

        self.target_upscale = self.create_target("UpscaleSkyAO")
        self.target_upscale.add_color_attachment(bits=(16, 0, 0, 0))
        self.target_upscale.prepare_buffer()

        self.target_upscale.set_shader_inputs(SourceTex=self.target.color_tex, upscaleWeights=Vec2(0.001, 0.001))

    def reload_shaders(self):
        self.target.shader = self.load_plugin_shader("compute_sky_ao.frag.glsl")
        self.target_upscale.shader = self.load_plugin_shader("/$$rp/shader/bilateral_upscale.frag.glsl")
