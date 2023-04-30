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

from panda3d.core import Vec3, Camera, OrthographicLens, PTAMat4
from panda3d.core import LVecBase2i

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.render_stage import RenderStage
from metadrive.render_pipeline.rpcore.util.generic import snap_shadow_map


class PSSMDistShadowStage(RenderStage):
    """ This stage generates a depth map using Variance Shadow Maps for very
    distant objects. """

    required_inputs = []

    def __init__(self, pipeline):
        RenderStage.__init__(self, pipeline)
        self.resolution = 2048
        self.clip_size = 500
        self.sun_distance = 8000
        self.sun_vector = Vec3(0, 0, 1)
        self.pta_mvp = PTAMat4.empty_array(1)

    @property
    def produced_inputs(self):
        return {"PSSMDistSunShadowMapMVP": self.pta_mvp}

    @property
    def produced_pipes(self):
        return {"PSSMDistSunShadowMap": self.target_blur_h.color_tex}

    @property
    def mvp(self):
        return Globals.base.render.get_transform(self.cam_node).get_mat() * \
            self.cam_lens.get_projection_mat()

    def update(self):
        self.target.active = False
        self.target_convert.active = False
        self.target_blur_v.active = False
        self.target_blur_h.active = False

        # Query scheduled tasks
        if self._pipeline.task_scheduler.is_scheduled("pssm_distant_shadows"):

            self.target.active = True

            # Reposition camera before we capture the scene
            cam_pos = Globals.base.cam.get_pos(Globals.base.render)
            self.cam_node.set_pos(cam_pos + self.sun_vector * self.sun_distance)
            self.cam_node.look_at(cam_pos)
            self.cam_lens.set_film_size(self.clip_size, self.clip_size)

            snap_shadow_map(self.mvp, self.cam_node, self.resolution)

        if self._pipeline.task_scheduler.is_scheduled("pssm_convert_distant_to_esm"):
            self.target_convert.active = True
        if self._pipeline.task_scheduler.is_scheduled("pssm_blur_distant_vert"):
            self.target_blur_v.active = True
        if self._pipeline.task_scheduler.is_scheduled("pssm_blur_distant_horiz"):
            self.target_blur_h.active = True

            # Only update the MVP as soon as the shadow map is available
            self.pta_mvp[0] = self.mvp

    def create(self):
        self.camera = Camera("PSSMDistShadowsESM")
        self.cam_lens = OrthographicLens()
        self.cam_lens.set_film_size(12000, 12000)
        self.cam_lens.set_near_far(10.0, self.sun_distance * 2)
        self.camera.set_lens(self.cam_lens)
        self.cam_node = Globals.base.render.attach_new_node(self.camera)

        self.target = self.create_target("ShadowMap")
        self.target.size = self.resolution
        self.target.add_depth_attachment(bits=32)
        self.target.prepare_render(self.cam_node)

        self.target_convert = self.create_target("ConvertToESM")
        self.target_convert.size = self.resolution
        self.target_convert.add_color_attachment(bits=(32, 0, 0, 0))
        self.target_convert.prepare_buffer()
        self.target_convert.set_shader_input("SourceTex", self.target.depth_tex)

        self.target_blur_v = self.create_target("BlurVert")
        self.target_blur_v.size = self.resolution
        self.target_blur_v.add_color_attachment(bits=(32, 0, 0, 0))
        self.target_blur_v.prepare_buffer()
        self.target_blur_v.set_shader_inputs(SourceTex=self.target_convert.color_tex, direction=LVecBase2i(1, 0))

        self.target_blur_h = self.create_target("BlurHoriz")
        self.target_blur_h.size = self.resolution
        self.target_blur_h.add_color_attachment(bits=(32, 0, 0, 0))
        self.target_blur_h.prepare_buffer()
        self.target_blur_h.set_shader_inputs(SourceTex=self.target_blur_v.color_tex, direction=LVecBase2i(0, 1))

        # Register shadow camera
        self._pipeline.tag_mgr.register_camera("shadow", self.camera)

    def reload_shaders(self):
        self.target_convert.shader = self.load_plugin_shader("convert_to_esm.frag.glsl")
        self.target_blur_v.shader = self.load_plugin_shader("blur_esm.frag.glsl")
        self.target_blur_h.shader = self.load_plugin_shader("blur_esm.frag.glsl")

    def set_shader_input(self, *args):
        Globals.render.set_shader_input(*args)

    def set_shader_inputs(self, **kwargs):
        Globals.render.set_shader_inputs(**kwargs)
