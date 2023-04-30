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


class MotionBlurStage(RenderStage):
    """ This stage applies the motion blur, using the screen space velocity vectors """

    required_inputs = []
    required_pipes = ["ShadedScene", "GBuffer", "DownscaledDepth", "CombinedVelocity"]

    @property
    def produced_pipes(self):
        return {"ShadedScene": self.target_cam_blur.color_tex}

    def create(self):

        if self.per_object_blur:
            self.tile_target = self.create_target("FetchVertDominantVelocity")
            self.tile_target.size = -1, -self.tile_size
            self.tile_target.add_color_attachment(bits=(16, 16, 0))
            self.tile_target.prepare_buffer()

            self.tile_target_horiz = self.create_target("FetchHorizDominantVelocity")
            self.tile_target_horiz.size = -self.tile_size
            self.tile_target_horiz.add_color_attachment(bits=(16, 16, 0))
            self.tile_target_horiz.prepare_buffer()
            self.tile_target_horiz.set_shader_input("SourceTex", self.tile_target.color_tex)

            self.minmax_target = self.create_target("NeighborMinMax")
            self.minmax_target.size = -self.tile_size
            self.minmax_target.add_color_attachment(bits=(16, 16, 0))
            self.minmax_target.prepare_buffer()
            self.minmax_target.set_shader_input("TileMinMax", self.tile_target_horiz.color_tex)

            self.pack_target = self.create_target("PackBlurData")
            self.pack_target.add_color_attachment(bits=(16, 16, 0))
            self.pack_target.prepare_buffer()

            self.target = self.create_target("ObjectMotionBlur")
            self.target.add_color_attachment(bits=16)
            self.target.prepare_buffer()
            self.target.set_shader_inputs(
                NeighborMinMax=self.minmax_target.color_tex, PackedSceneData=self.pack_target.color_tex
            )

            self.target.color_tex.set_wrap_u(SamplerState.WM_clamp)
            self.target.color_tex.set_wrap_v(SamplerState.WM_clamp)

        self.target_cam_blur = self.create_target("CameraMotionBlur")
        self.target_cam_blur.add_color_attachment(bits=16)
        self.target_cam_blur.prepare_buffer()

        if self.per_object_blur:
            self.target_cam_blur.set_shader_input("SourceTex", self.target.color_tex)

    def reload_shaders(self):
        if self.per_object_blur:
            self.tile_target.shader = self.load_plugin_shader("fetch_dominant_velocity.frag.glsl")
            self.tile_target_horiz.shader = self.load_plugin_shader("fetch_dominant_velocity_horiz.frag.glsl")
            self.minmax_target.shader = self.load_plugin_shader("neighbor_minmax.frag.glsl")
            self.pack_target.shader = self.load_plugin_shader("pack_blur_data.frag.glsl")
            self.target.shader = self.load_plugin_shader("apply_motion_blur.frag.glsl")
        self.target_cam_blur.shader = self.load_plugin_shader("camera_motion_blur.frag.glsl")
