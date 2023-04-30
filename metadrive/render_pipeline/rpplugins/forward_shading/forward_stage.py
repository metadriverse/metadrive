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

from panda3d.core import Camera

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.render_stage import RenderStage


class ForwardStage(RenderStage):
    """ Forward shading stage, which first renders all forward objects,
    and then merges them with the scene """

    required_inputs = ["DefaultEnvmap", "PrefilteredBRDF", "PrefilteredCoatBRDF"]
    required_pipes = ["SceneDepth", "ShadedScene", "CellIndices"]

    @property
    def produced_pipes(self):
        return {"ShadedScene": self.target_merge.color_tex}

    def create(self):
        self.forward_cam = Camera("ForwardShadingCam")
        self.forward_cam.set_lens(Globals.base.camLens)
        self.forward_cam_np = Globals.base.camera.attach_new_node(self.forward_cam)

        self.target = self.create_target("ForwardShading")
        self.target.add_color_attachment(bits=16, alpha=True)
        self.target.add_depth_attachment(bits=32)
        self.target.prepare_render(self.forward_cam_np)
        self.target.set_clear_color(0, 0, 0, 0)

        self._pipeline.tag_mgr.register_camera("forward", self.forward_cam)

        self.target_merge = self.create_target("MergeWithDeferred")
        self.target_merge.add_color_attachment(bits=16)
        self.target_merge.prepare_buffer()
        self.target_merge.set_shader_inputs(ForwardDepth=self.target.depth_tex, ForwardColor=self.target.color_tex)

    def set_shader_input(self, *args):
        Globals.base.render.set_shader_input(*args)
        RenderStage.set_shader_input(self, *args)

    def set_shader_inputs(self, **kwargs):
        Globals.base.render.set_shader_inputs(**kwargs)
        RenderStage.set_shader_inputs(self, **kwargs)

    def reload_shaders(self):
        self.target_merge.shader = self.load_plugin_shader("merge_with_deferred.frag.glsl")
