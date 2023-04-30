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

from panda3d.core import Vec3, Camera, OrthographicLens, PTAMat4, SamplerState

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.render_stage import RenderStage
from metadrive.render_pipeline.rpcore.util.generic import snap_shadow_map


class PSSMSceneShadowStage(RenderStage):
    """ This stage creates the shadow map which covers the whole important part
    of the scene. This is required because the shadow cascades only cover the
    view frustum, but many plugins (VXGI, EnvMaps) require a shadow map. """

    required_inputs = []
    required_pipes = []

    def __init__(self, pipeline):
        RenderStage.__init__(self, pipeline)
        self.resolution = 2048
        self.sun_vector = Vec3(0, 0, 1)
        self.sun_distance = 10.0
        self.pta_mvp = PTAMat4.empty_array(1)
        self.focus = None

        # Store last focus entirely for the purpose of being able to see
        # it in the debugger
        self.last_focus = None

    @property
    def produced_inputs(self):
        return {"PSSMSceneSunShadowMVP": self.pta_mvp}

    @property
    def produced_pipes(self):
        return {"PSSMSceneSunShadowMapPCF": (self.target.depth_tex, self.make_pcf_state())}

    def make_pcf_state(self):
        state = SamplerState()
        state.set_minfilter(SamplerState.FT_shadow)
        state.set_magfilter(SamplerState.FT_shadow)
        return state

    def request_focus(self, focus_point, focus_size):
        self.focus = (focus_point, focus_size)
        self.last_focus = self.focus

    @property
    def mvp(self):
        return Globals.base.render.get_transform(self.cam_node).get_mat() * \
            self.cam_lens.get_projection_mat()

    def update(self):
        if self._pipeline.task_scheduler.is_scheduled("pssm_scene_shadows"):
            if self.focus is None:
                # When no focus is set, there is no point in rendering the shadow map
                self.target.active = False
            else:
                focus_point, focus_size = self.focus

                self.cam_lens.set_near_far(0.0, 2 * (focus_size + self.sun_distance))
                self.cam_lens.set_film_size(2 * focus_size, 2 * focus_size)
                self.cam_node.set_pos(focus_point + self.sun_vector * (self.sun_distance + focus_size))
                self.cam_node.look_at(focus_point)

                snap_shadow_map(self.mvp, self.cam_node, self.resolution)
                self.target.active = True
                self.pta_mvp[0] = self.mvp

                self.focus = None
        else:
            self.target.active = False

    def create(self):
        self.camera = Camera("PSSMSceneSunShadowCam")
        self.cam_lens = OrthographicLens()
        self.cam_lens.set_film_size(400, 400)
        self.cam_lens.set_near_far(100.0, 1800.0)
        self.camera.set_lens(self.cam_lens)
        self.cam_node = Globals.base.render.attach_new_node(self.camera)

        self.target = self.create_target("ShadowMap")
        self.target.size = self.resolution
        self.target.add_depth_attachment(bits=32)
        self.target.prepare_render(self.cam_node)

        # Register shadow camera
        self._pipeline.tag_mgr.register_camera("shadow", self.camera)

    def set_shader_input(self, *args):
        Globals.render.set_shader_input(*args)

    def set_shader_inputs(self, **kwargs):
        Globals.render.set_shader_inputs(**kwargs)
