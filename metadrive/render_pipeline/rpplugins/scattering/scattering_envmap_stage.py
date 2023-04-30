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

from metadrive.render_pipeline.rpcore.render_stage import RenderStage
from metadrive.render_pipeline.rpcore.util.cubemap_filter import CubemapFilter

from metadrive.render_pipeline.rpcore.stages.ambient_stage import AmbientStage
from metadrive.render_pipeline.rpcore.stages.gbuffer_stage import GBufferStage


class ScatteringEnvmapStage(RenderStage):
    """ This stage uses the precomputed data to make a cubemap containing the
    scattering """

    required_pipes = []
    required_inputs = ["DefaultSkydome", "DefaultEnvmap"]

    @property
    def produced_pipes(self):
        return {
            "ScatteringIBLDiffuse": self.cubemap_filter.diffuse_cubemap,
            "ScatteringIBLSpecular": self.cubemap_filter.specular_cubemap,
        }

    def create(self):
        self.cubemap_filter = CubemapFilter(self, "ScatEnvCub")

        self.target_cube = self.create_target("ComputeScattering")
        self.target_cube.size = self.cubemap_filter.size * 6, self.cubemap_filter.size
        self.target_cube.prepare_buffer()
        self.target_cube.set_shader_input("DestCubemap", self.cubemap_filter.target_cubemap)

        self.cubemap_filter.create()

        # Make the stages use our cubemap textures
        for stage in (AmbientStage, GBufferStage):
            stage.required_pipes.append("ScatteringIBLDiffuse")
            stage.required_pipes.append("ScatteringIBLSpecular")

    def reload_shaders(self):
        self.target_cube.shader = self.load_plugin_shader("scattering_envmap.frag.glsl")
        self.cubemap_filter.reload_shaders()
