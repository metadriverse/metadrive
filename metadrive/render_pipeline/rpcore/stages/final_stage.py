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


class FinalStage(RenderStage):
    """ This stage is the final stage and outputs the shaded scene to the
    screen """

    required_pipes = ["ShadedScene"]

    @property
    def produced_pipes(self):
        return {"ShadedScene": self.target.color_tex}

    def create(self):

        self.target = self.create_target("FinalStage")
        self.target.add_color_attachment(bits=16)
        self.target.prepare_buffer()

        # XXX: We cannot simply assign the final shader to the window display
        # region. This is because of a bug that the last FBO always only gets
        # 8 bits, regardles of what was requested - probably because of the
        # assumption that no tonemapping/srgb correction will follow afterwards.
        #
        # This might be a driver bug, or an optimization in Panda3D. However, it
        # also has the nice side effect that when taking screenshots (or using
        # the pixel inspector), we get the srgb corrected data, so its not too
        # much of a disadvantage.
        self.present_target = self.create_target("FinalPresentStage")
        self.present_target.present_on_screen()
        self.present_target.set_shader_input("SourceTex", self.target.color_tex)

    def reload_shaders(self):
        self.target.shader = self.load_shader("final_stage.frag.glsl")
        self.present_target.shader = self.load_shader("final_present_stage.frag.glsl")
