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

from panda3d.core import Vec4
from metadrive.render_pipeline.rpcore.render_stage import RenderStage


class PSSMStage(RenderStage):
    """ This stage uses the PSSM Shadow map to render the shadows """

    required_inputs = []
    required_pipes = ["ShadedScene", "PSSMShadowAtlas", "GBuffer", "PSSMShadowAtlasPCF"]

    @property
    def produced_pipes(self):
        return {"ShadedScene": self.target.color_tex}

    def create(self):
        self.enabled = True
        self.target_shadows = self.create_target("FilterPSSM")
        self.target_shadows.add_color_attachment(bits=(8, 0, 0, 0))
        self.target_shadows.prepare_buffer()
        self.target_shadows.color_tex.set_clear_color(Vec4(0))

        self.target = self.create_target("ApplyPSSMShadows")
        self.target.add_color_attachment(bits=16)
        self.target.prepare_buffer()

        self.target.set_shader_input("PrefilteredShadows", self.target_shadows.color_tex)

    def set_render_shadows(self, enabled):
        """ Toggle whether to render shadows or whether to just pass through
        the scene color """
        self.target_shadows.active = enabled
        if enabled != self.enabled:
            self.target_shadows.color_tex.clear_image()
            self.enabled = enabled

    def reload_shaders(self):
        self.target_shadows.shader = self.load_plugin_shader("filter_pssm_shadows.frag.glsl")
        self.target.shader = self.load_plugin_shader("apply_sun_shading.frag.glsl")
