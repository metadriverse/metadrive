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


class VolumetricsStage(RenderStage):
    """ This stage applies the volumetric lighting """

    required_inputs = []
    required_pipes = ["ShadedScene", "GBuffer"]

    def __init__(self, pipeline):
        RenderStage.__init__(self, pipeline)
        self.enable_volumetric_shadows = False

    @property
    def produced_pipes(self):
        return {"ShadedScene": self.target_combine.color_tex}

    def create(self):

        if self.enable_volumetric_shadows:
            self.target = self.create_target("ComputeVolumetrics")
            self.target.size = -2
            self.target.add_color_attachment(bits=16, alpha=True)
            self.target.prepare_buffer()

            self.target_upscale = self.create_target("Upscale")
            self.target_upscale.add_color_attachment(bits=16, alpha=True)
            self.target_upscale.prepare_buffer()

            self.target_upscale.set_shader_inputs(SourceTex=self.target.color_tex, upscaleWeights=Vec2(0.001, 0.001))

        self.target_combine = self.create_target("CombineVolumetrics")
        self.target_combine.add_color_attachment(bits=16)
        self.target_combine.prepare_buffer()

        if self.enable_volumetric_shadows:
            self.target_combine.set_shader_input("VolumetricsTex", self.target_upscale.color_tex)

    def reload_shaders(self):
        if self.enable_volumetric_shadows:
            self.target.shader = self.load_plugin_shader("compute_volumetric_shadows.frag.glsl")
            self.target_upscale.shader = self.load_plugin_shader("/$$rp/shader/bilateral_upscale.frag.glsl")
        self.target_combine.shader = self.load_plugin_shader("apply_volumetrics.frag.glsl")
