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

# Load the plugin api
from metadrive.render_pipeline.rpcore.pluginbase.base_plugin import BasePlugin
from metadrive.render_pipeline.rpcore.loader import RPLoader

from .color_correction_stage import ColorCorrectionStage
from .auto_exposure_stage import AutoExposureStage
from .manual_exposure_stage import ManualExposureStage
from .sharpen_stage import SharpenStage
from .tonemapping_stage import TonemappingStage


class Plugin(BasePlugin):

    name = "Color Correction"
    author = "tobspr <tobias.springer1@gmail.com>"
    description = (
        "This plugin adds support for color correction, vignetting, "
        "chromatic abberation and tonemapping. It also controls the "
        "camera parameters."
    )
    version = "1.4"

    def on_stage_setup(self):
        self.stage = self.create_stage(ColorCorrectionStage)
        self.tonemapping_stage = self.create_stage(TonemappingStage)

        if self.get_setting("use_sharpen"):
            self.sharpen_stage = self.create_stage(SharpenStage)
            self.sharpen_stage.sharpen_twice = self.get_setting("sharpen_twice")

        if not self.get_setting("manual_camera_parameters"):
            self.exposure_stage = self.create_stage(AutoExposureStage)
        else:
            self.exposure_stage = self.create_stage(ManualExposureStage)  # noqa # pylint: disable=redefined-variable-type

    def on_pipeline_created(self):
        self.load_lut()
        self.load_grain()

    def load_lut(self):
        """ Loads the color correction lookup table (LUT) """
        lut_path = self.get_resource(self.get_setting("color_lut"))
        lut = RPLoader.load_sliced_3d_texture(lut_path, 64)
        lut.set_wrap_u(SamplerState.WM_clamp)
        lut.set_wrap_v(SamplerState.WM_clamp)
        lut.set_wrap_w(SamplerState.WM_clamp)
        lut.set_minfilter(SamplerState.FT_linear)
        lut.set_magfilter(SamplerState.FT_linear)
        lut.set_anisotropic_degree(0)
        self.tonemapping_stage.set_shader_input("ColorLUT", lut)

    def load_grain(self):
        """ Loads the precomputed film grain """
        grain_tex = RPLoader.load_texture("/$$rp/data/film_grain/grain.txo")
        grain_tex.set_minfilter(SamplerState.FT_linear)
        grain_tex.set_magfilter(SamplerState.FT_linear)
        grain_tex.set_wrap_u(SamplerState.WM_repeat)
        grain_tex.set_wrap_v(SamplerState.WM_repeat)
        grain_tex.set_anisotropic_degree(0)
        self.stage.set_shader_input("PrecomputedGrain", grain_tex)

    def update_color_lut(self):
        self.debug("Updating color lut ..")
        self.load_lut()
