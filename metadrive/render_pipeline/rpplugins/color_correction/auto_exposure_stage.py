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
from panda3d.core import Vec4

from metadrive.render_pipeline.rpcore.render_stage import RenderStage
from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.image import Image


class AutoExposureStage(RenderStage):

    required_pipes = ["ShadedScene"]
    required_inputs = []

    @property
    def produced_pipes(self):
        return {"ShadedScene": self.target_apply.color_tex, "Exposure": self.tex_exposure}

    def create(self):

        # Create the target which converts the scene color to a luminance
        self.target_lum = self.create_target("GetLuminance")
        self.target_lum.size = -4
        self.target_lum.add_color_attachment(bits=(16, 0, 0, 0))
        self.target_lum.prepare_buffer()

        self.mip_targets = []

        # Create the storage for the exposure, this stores the current and last
        # frames exposure
        # XXX: We have to use F_r16 instead of F_r32 because of a weird nvidia
        # driver bug! However, 16 bits should be enough for sure.
        self.tex_exposure = Image.create_buffer("ExposureStorage", 1, "R16")
        self.tex_exposure.set_clear_color(Vec4(0.5))
        self.tex_exposure.clear_image()

        # Create the target which extracts the exposure from the average brightness
        self.target_analyze = self.create_target("AnalyzeBrightness")
        self.target_analyze.size = 1, 1
        self.target_analyze.prepare_buffer()

        self.target_analyze.set_shader_input("ExposureStorage", self.tex_exposure)

        # Create the target which applies the generated exposure to the scene
        self.target_apply = self.create_target("ApplyExposure")
        self.target_apply.add_color_attachment(bits=16)
        self.target_apply.prepare_buffer()
        self.target_apply.set_shader_input("Exposure", self.tex_exposure)

    def set_dimensions(self):
        for old_target in self.mip_targets:
            self.remove_target(old_target)

        wsize_x = (Globals.resolution.x + 3) // 4
        wsize_y = (Globals.resolution.y + 3) // 4

        # Create the targets which downscale the luminance mipmaps
        self.mip_targets = []
        last_tex = self.target_lum.color_tex
        while wsize_x >= 4 or wsize_y >= 4:
            wsize_x = (wsize_x + 3) // 4
            wsize_y = (wsize_y + 3) // 4

            mip_target = self.create_target("DScaleLum:S" + str(wsize_x))
            mip_target.add_color_attachment(bits=(16, 0, 0, 0))
            mip_target.size = wsize_x, wsize_y
            mip_target.sort = self.target_lum.sort + len(self.mip_targets)
            mip_target.prepare_buffer()
            mip_target.set_shader_input("SourceTex", last_tex)
            self.mip_targets.append(mip_target)
            last_tex = mip_target.color_tex

        self.target_analyze.set_shader_input("DownscaledTex", self.mip_targets[-1].color_tex)

        # Shaders might not have been loaded at this point
        if hasattr(self, "mip_shader"):
            for target in self.mip_targets:
                target.shader = self.mip_shader

    def reload_shaders(self):
        self.target_lum.shader = self.load_plugin_shader("generate_luminance.frag.glsl")
        self.target_analyze.shader = self.load_plugin_shader("analyze_brightness.frag.glsl")
        self.target_apply.shader = self.load_plugin_shader("apply_exposure.frag.glsl")

        # Keep shader as reference, required when resizing
        self.mip_shader = self.load_plugin_shader("downscale_luminance.frag.glsl")
        for target in self.mip_targets:
            target.shader = self.mip_shader
