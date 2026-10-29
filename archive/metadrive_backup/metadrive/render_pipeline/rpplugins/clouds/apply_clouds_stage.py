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


class ApplyCloudsStage(RenderStage):
    """ This stage raymarchs the cloud voxel grid """

    required_pipes = ["ShadedScene", "GBuffer"]

    def __init__(self, pipeline):
        RenderStage.__init__(self, pipeline)

    @property
    def produced_pipes(self):
        return {"ShadedScene": self.target_apply_clouds.color_tex}

    def create(self):
        self.render_target = self.create_target("RaymarchVoxels")
        self.render_target.size = -2
        self.render_target.add_color_attachment(bits=16, alpha=True)
        self.render_target.prepare_buffer()

        self.upscale_target = self.create_target("UpscaleTarget")
        self.upscale_target.add_color_attachment(bits=16, alpha=True)
        self.upscale_target.prepare_buffer()
        self.upscale_target.set_shader_inputs(upscaleWeights=Vec2(0.05, 0.2), SourceTex=self.render_target.color_tex)

        self.target_apply_clouds = self.create_target("MergeWithScene")
        self.target_apply_clouds.add_color_attachment(bits=16)
        self.target_apply_clouds.prepare_buffer()

        self.target_apply_clouds.set_shader_input("CloudsTex", self.upscale_target.color_tex)

    def reload_shaders(self):
        self.target_apply_clouds.shader = self.load_plugin_shader("apply_clouds.frag.glsl")
        self.render_target.shader = self.load_plugin_shader("render_clouds.frag.glsl")
        self.upscale_target.shader = self.load_plugin_shader("/$$rp/shader/bilateral_upscale.frag.glsl")
