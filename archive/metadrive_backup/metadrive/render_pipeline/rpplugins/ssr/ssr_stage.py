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

from panda3d.core import SamplerState

from metadrive.render_pipeline.rpcore.render_stage import RenderStage
from metadrive.render_pipeline.rpcore.stages.ambient_stage import AmbientStage


class SSRStage(RenderStage):
    """ This stage does the SSR pass """

    required_inputs = []
    required_pipes = [
        "ShadedScene", "CombinedVelocity", "GBuffer", "DownscaledDepth", "PreviousFrame::PostAmbientScene",
        "PreviousFrame::SSRSpecular", "PreviousFrame::SceneDepth"
    ]

    @property
    def produced_pipes(self):
        return {"SSRSpecular": self.target_resolve.color_tex}

    def create(self):
        self.target = self.create_target("ComputeSSR")
        self.target.size = -2
        self.target.add_color_attachment(bits=(16, 16, 0, 0))
        self.target.prepare_buffer()

        self.target.color_tex.set_minfilter(SamplerState.FT_nearest)
        self.target.color_tex.set_magfilter(SamplerState.FT_nearest)

        self.target_velocity = self.create_target("ReflectionVelocity")
        self.target_velocity.add_color_attachment(bits=(16, 16, 0, 0))
        self.target_velocity.prepare_buffer()
        self.target_velocity.set_shader_input("TraceResult", self.target.color_tex)

        self.target_reproject_lighting = self.create_target("CopyLighting")
        self.target_reproject_lighting.add_color_attachment(bits=16, alpha=True)
        self.target_reproject_lighting.prepare_buffer()

        self.target_upscale = self.create_target("UpscaleSSR")
        self.target_upscale.add_color_attachment(bits=16, alpha=True)
        self.target_upscale.prepare_buffer()
        self.target_upscale.set_shader_inputs(
            SourceTex=self.target.color_tex, LastFrameColor=self.target_reproject_lighting.color_tex
        )

        self.target_resolve = self.create_target("ResolveSSR")
        self.target_resolve.add_color_attachment(bits=16, alpha=True)
        self.target_resolve.prepare_buffer()
        self.target_resolve.set_shader_inputs(
            CurrentTex=self.target_upscale.color_tex, VelocityTex=self.target_velocity.color_tex
        )

        AmbientStage.required_pipes.append("SSRSpecular")

    def reload_shaders(self):
        self.target.shader = self.load_plugin_shader("ssr_trace.frag.glsl")
        self.target_velocity.shader = self.load_plugin_shader("reflection_velocity.frag.glsl")
        self.target_reproject_lighting.shader = self.load_plugin_shader("reproject_lighting.frag.glsl")
        self.target_upscale.shader = self.load_plugin_shader("upscale_bilateral_brdf.frag.glsl")
        self.target_resolve.shader = self.load_plugin_shader("resolve_ssr.frag.glsl")
