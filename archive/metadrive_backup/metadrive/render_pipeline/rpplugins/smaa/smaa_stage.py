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

from panda3d.core import PTAInt

from metadrive.render_pipeline.rpcore.render_stage import RenderStage


class SMAAStage(RenderStage):
    """ This stage does the actual SMAA """

    required_inputs = []

    def __init__(self, pipeline):
        RenderStage.__init__(self, pipeline)
        self.area_tex = None
        self.search_tex = None
        self.use_reprojection = True
        self._jitter_index = PTAInt.empty_array(1)

    def set_jitter_index(self, idx):
        """ Sets the current jitter index """
        self._jitter_index[0] = idx

    @property
    def required_pipes(self):
        pipes = ["ShadedScene", "GBuffer", "CombinedVelocity"]
        if self.use_reprojection:
            pipes.append("PreviousFrame::SMAAPostResolve")
        return pipes

    @property
    def produced_pipes(self):
        if self.use_reprojection:
            return {"ShadedScene": self.resolve_target.color_tex, "SMAAPostResolve": self.resolve_target.color_tex}
        else:
            return {"ShadedScene": self.neighbor_target.color_tex}

    def create(self):
        # Edge detection
        self.edge_target = self.create_target("EdgeDetection")
        self.edge_target.add_color_attachment()
        self.edge_target.prepare_buffer()
        self.edge_target.set_clear_color(0, 0, 0, 0)

        # Weight blending
        self.blend_target = self.create_target("BlendWeights")
        self.blend_target.add_color_attachment(alpha=True)
        self.blend_target.prepare_buffer()

        self.blend_target.set_shader_inputs(
            EdgeTex=self.edge_target.color_tex,
            AreaTex=self.area_tex,
            SearchTex=self.search_tex,
            jitterIndex=self._jitter_index
        )

        # Neighbor blending
        self.neighbor_target = self.create_target("NeighborBlending")
        self.neighbor_target.add_color_attachment(bits=16)
        self.neighbor_target.prepare_buffer()
        self.neighbor_target.set_shader_input("BlendTex", self.blend_target.color_tex)

        # Resolving
        if self.use_reprojection:
            self.resolve_target = self.create_target("Resolve")
            self.resolve_target.add_color_attachment(bits=16)
            self.resolve_target.prepare_buffer()
            self.resolve_target.set_shader_inputs(
                jitterIndex=self._jitter_index,
                # Set initial textures
                CurrentTex=self.neighbor_target.color_tex
            )

    def reload_shaders(self):
        self.edge_target.shader = self.load_plugin_shader("edge_detection.frag.glsl")
        self.blend_target.shader = self.load_plugin_shader("blending_weights.frag.glsl")
        self.neighbor_target.shader = self.load_plugin_shader("neighborhood_blending.frag.glsl")
        if self.use_reprojection:
            self.resolve_target.shader = self.load_plugin_shader("resolve_smaa.frag.glsl")
