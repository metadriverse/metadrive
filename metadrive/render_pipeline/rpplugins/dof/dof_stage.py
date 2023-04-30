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

from metadrive.render_pipeline.rpcore.render_stage import RenderStage


class DoFStage(RenderStage):
    """ This stage does the DoF pass """

    required_inputs = []
    required_pipes = ["ShadedScene", "GBuffer", "DownscaledDepth"]

    @property
    def produced_pipes(self):
        return {"ShadedScene": self.target_merge.color_tex}

    def create(self):

        self.target_prefilter = self.create_target("PrefilterDoF")
        # self.target_prefilter.size = -2
        self.target_prefilter.add_color_attachment(bits=16, alpha=True)
        self.target_prefilter.prepare_buffer()

        self.tile_size = 32
        self.tile_target = self.create_target("FetchVertDOF")
        self.tile_target.size = -1, -self.tile_size
        self.tile_target.add_color_attachment(bits=(16, 16, 0))
        self.tile_target.prepare_buffer()

        self.tile_target.set_shader_input("PrecomputedCoC", self.target_prefilter.color_tex)

        self.tile_target_horiz = self.create_target("FetchHorizDOF")
        self.tile_target_horiz.size = -self.tile_size
        self.tile_target_horiz.add_color_attachment(bits=(16, 16, 0))
        self.tile_target_horiz.prepare_buffer()
        self.tile_target_horiz.set_shader_input("SourceTex", self.tile_target.color_tex)

        self.minmax_target = self.create_target("DoFNeighborMinMax")
        self.minmax_target.size = -self.tile_size
        self.minmax_target.add_color_attachment(bits=(16, 16, 0))
        self.minmax_target.prepare_buffer()
        self.minmax_target.set_shader_input("TileMinMax", self.tile_target_horiz.color_tex)

        self.presort_target = self.create_target("DoFPresort")
        self.presort_target.add_color_attachment(bits=(11, 11, 10))
        self.presort_target.prepare_buffer()
        self.presort_target.set_shader_inputs(
            TileMinMax=self.minmax_target.color_tex, PrecomputedCoC=self.target_prefilter.color_tex
        )

        self.target = self.create_target("ComputeDoF")
        # self.target.size = -2
        self.target.add_color_attachment(bits=16, alpha=True)
        self.target.prepare_buffer()
        self.target.set_shader_inputs(
            PresortResult=self.presort_target.color_tex,
            PrecomputedCoC=self.target_prefilter.color_tex,
            TileMinMax=self.minmax_target.color_tex
        )

        self.target_merge = self.create_target("MergeDoF")
        self.target_merge.add_color_attachment(bits=16)
        self.target_merge.prepare_buffer()
        self.target_merge.set_shader_input("SourceTex", self.target.color_tex)

        # self.target_upscale.set_shader_input("SourceTex", self.target.color_tex)
        # self.target_upscale.set_shader_input("upscaleWeights", Vec2(0.001, 0.001))

    def reload_shaders(self):
        self.tile_target.shader = self.load_plugin_shader("fetch_dof_minmax.frag.glsl")
        self.tile_target_horiz.shader = self.load_plugin_shader("fetch_dof_minmax_horiz.frag.glsl")
        self.minmax_target.shader = self.load_plugin_shader("fetch_dof_tile_neighbors.frag.glsl")
        self.presort_target.shader = self.load_plugin_shader("dof_presort.frag.glsl")
        self.target_prefilter.shader = self.load_plugin_shader("prefilter_dof.frag.glsl")
        self.target.shader = self.load_plugin_shader("compute_dof.frag.glsl")
        self.target_merge.shader = self.load_plugin_shader("merge_dof.frag.glsl")
        # self.target_upscale.shader = self.load_plugin_shader(
        #     "/$$rp/shader/bilateral_upscale.frag.glsl")
