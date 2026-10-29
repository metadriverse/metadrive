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

from panda3d.core import Camera, OrthographicLens, PTALVecBase3f

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.render_stage import RenderStage


class SkyAOCaptureStage(RenderStage):
    """ This stage captures the sky ao by rendering the scene from above """

    required_inputs = []
    required_pipes = []

    @property
    def produced_pipes(self):
        return {"SkyAOHeight": self.target_convert.color_tex}

    @property
    def produced_inputs(self):
        return {"SkyAOCapturePosition": self.pta_position}

    def __init__(self, pipeline):
        RenderStage.__init__(self, pipeline)
        self.pta_position = PTALVecBase3f.empty_array(1)
        self.resolution = 512
        self.capture_height = 100.0
        self.max_radius = 100.0

    def create(self):
        self.camera = Camera("SkyAOCaptureCam")
        self.cam_lens = OrthographicLens()
        self.cam_lens.set_film_size(self.max_radius, self.max_radius)
        self.cam_lens.set_near_far(0, self.capture_height)
        self.camera.set_lens(self.cam_lens)
        self.cam_node = Globals.base.render.attach_new_node(self.camera)
        self.cam_node.look_at(0, 0, -1)
        self.cam_node.set_r(0)

        self.target = self.create_target("SkyAOCapture")
        self.target.size = self.resolution
        self.target.add_depth_attachment(bits=16)
        self.target.prepare_render(self.cam_node)

        self.target_convert = self.create_target("ConvertDepth")
        self.target_convert.size = self.resolution
        self.target_convert.add_color_attachment(bits=(16, 0, 0, 0))
        self.target_convert.prepare_buffer()

        self.target_convert.set_shader_inputs(DepthTex=self.target.depth_tex, position=self.pta_position)

        # Register camera
        self._pipeline.tag_mgr.register_camera("shadow", self.camera)

    def update(self):
        snap_size = self.max_radius / self.resolution
        cam_pos = Globals.base.camera.get_pos(Globals.base.render)
        self.cam_node.set_pos(
            cam_pos.x - cam_pos.x % snap_size, cam_pos.y - cam_pos.y % snap_size, self.capture_height / 2.0
        )
        self.pta_position[0] = self.cam_node.get_pos()

    def reload_shaders(self):
        self.target_convert.shader = self.load_plugin_shader("convert_depth.frag.glsl")
