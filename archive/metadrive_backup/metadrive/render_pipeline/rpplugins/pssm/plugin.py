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
from __future__ import print_function

from metadrive.render_pipeline.rplibs.six.moves import range  # pylint: disable=import-error
from panda3d.core import PTAVecBase3f

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.pluginbase.base_plugin import BasePlugin
from metadrive.render_pipeline.rpcore.native import PSSMCameraRig, NATIVE_CXX_LOADED

from .pssm_stage import PSSMStage
from .pssm_shadow_stage import PSSMShadowStage
from .pssm_scene_shadow_stage import PSSMSceneShadowStage
from .pssm_dist_shadow_stage import PSSMDistShadowStage


class Plugin(BasePlugin):

    name = "Sun Shadows"
    author = "tobspr <tobias.springer1@gmail.com>"
    description = ("This plugin adds support for Parallel Split Shadow Maps " "(PSSM), and also sun lighting.")
    version = "1.2"
    required_plugins = ("scattering", )

    def on_stage_setup(self):

        if not NATIVE_CXX_LOADED:
            self.debug("Setting max splits to 1 since python is used")
            self._pipeline.plugin_mgr.settings["pssm"]["split_count"].set_value(1)

        self.update_enabled = True
        self.pta_sun_vector = PTAVecBase3f.empty_array(1)
        self.last_cache_reset = 0

        self.shadow_stage = self.create_stage(PSSMShadowStage)
        self.pssm_stage = self.create_stage(PSSMStage)

        self.shadow_stage.num_splits = self.get_setting("split_count")
        self.shadow_stage.split_resolution = self.get_setting("resolution")

        self.scene_shadow_stage = self.create_stage(PSSMSceneShadowStage)
        self.scene_shadow_stage.resolution = self.get_setting("scene_shadow_resolution")
        self.scene_shadow_stage.sun_distance = self.get_setting("scene_shadow_sundist")

        # Enable distant shadow map if specified
        if self.get_setting("use_distant_shadows"):
            self.dist_shadow_stage = self.create_stage(PSSMDistShadowStage)
            self.dist_shadow_stage.resolution = self.get_setting("dist_shadow_resolution")
            self.dist_shadow_stage.clip_size = self.get_setting("dist_shadow_clipsize")
            self.dist_shadow_stage.sun_distance = self.get_setting("dist_shadow_sundist")

            self.pssm_stage.required_pipes.append("PSSMDistSunShadowMap")
            self.pssm_stage.required_inputs.append("PSSMDistSunShadowMapMVP")

    def on_pipeline_created(self):
        self.debug("Initializing pssm ..")
        # Construct a dummy node to parent the rig to
        self.node = Globals.base.render.attach_new_node("PSSMCameraRig")
        self.node.hide()

        # Construct the actual PSSM rig
        self.camera_rig = PSSMCameraRig(self.get_setting("split_count"))  # noqa # pylint: disable=undefined-variable
        self.camera_rig.set_sun_distance(self.get_setting("sun_distance"))
        self.camera_rig.set_pssm_distance(self.get_setting("max_distance"))
        self.camera_rig.set_logarithmic_factor(self.get_setting("logarithmic_factor"))
        self.camera_rig.set_border_bias(self.get_setting("border_bias"))
        self.camera_rig.set_use_stable_csm(True)
        self.camera_rig.set_use_fixed_film_size(True)
        self.camera_rig.set_resolution(self.get_setting("resolution"))
        self.camera_rig.reparent_to(self.node)

        # Attach the cameras to the shadow stage
        for i in range(self.get_setting("split_count")):
            camera_np = self.camera_rig.get_camera(i)
            camera_np.node().set_scene(Globals.base.render)
            region = self.shadow_stage.split_regions[i]
            region.set_camera(camera_np)
            # camera_np.node().show_frustum()

            # Make sure the pipeline knows about our camera, so it can apply
            # the correct bitmasks
            self._pipeline.tag_mgr.register_camera("shadow", camera_np.node())

        # Accept a shortcut to enable / disable the update of PSSM
        Globals.base.accept("u", self.toggle_update_enabled)

        # Set inputs
        self.pssm_stage.set_shader_inputs(
            pssm_mvps=self.camera_rig.get_mvp_array(), pssm_nearfar=self.camera_rig.get_nearfar_array()
        )

        if self.is_plugin_enabled("volumetrics"):
            handle = self.get_plugin_instance("volumetrics")
            handle.stage.set_shader_inputs(
                pssm_mvps=self.camera_rig.get_mvp_array(), pssm_nearfar=self.camera_rig.get_nearfar_array()
            )

    def on_pre_render_update(self):
        sun_vector = self.get_plugin_instance("scattering").sun_vector

        if sun_vector.z < 0.0:
            self.shadow_stage.active = False
            self.scene_shadow_stage.active = False
            self.pssm_stage.set_render_shadows(False)

            if self.get_setting("use_distant_shadows"):
                self.dist_shadow_stage.active = False

            # Return, no need to update the pssm splits
            return
        else:
            self.shadow_stage.active = True
            self.scene_shadow_stage.active = True
            self.pssm_stage.set_render_shadows(True)

            if self.get_setting("use_distant_shadows"):
                self.dist_shadow_stage.active = True

        if self.update_enabled:
            self.camera_rig.update(Globals.base.camera, sun_vector)

            # Eventually reset cache
            cache_diff = Globals.clock.get_frame_time() - self.last_cache_reset
            if cache_diff > 5.0:
                self.last_cache_reset = Globals.clock.get_frame_time()
                self.camera_rig.reset_film_size_cache()

            self.scene_shadow_stage.sun_vector = sun_vector

            if self.get_setting("use_distant_shadows"):
                self.dist_shadow_stage.sun_vector = sun_vector

    def update_max_distance(self):
        self.camera_rig.set_pssm_distance(self.get_setting("max_distance"))

    def update_logarithmic_factor(self):
        self.camera_rig.set_logarithmic_factor(self.get_setting("logarithmic_factor"))

    def update_border_bias(self):
        self.camera_rig.set_border_bias(self.get_setting("border_bias"))

    def update_sun_distance(self):
        self.camera_rig.set_sun_distance(self.get_setting("sun_distance"))

    def update_dist_shadow_clipsize(self):
        self.dist_shadow_stage.clip_size = self.get_setting("dist_shadow_clipsize")

    def update_dist_shadow_sundist(self):
        self.dist_shadow_stage.sun_distance = self.get_setting("dist_shadow_sundist")

    def update_scene_shadow_sundist(self):
        self.scene_shadow_stage.sun_distance = self.get_setting("scene_shadow_sundist")

    def toggle_update_enabled(self):
        self.update_enabled = not self.update_enabled
        self.debug("Update enabled:", self.update_enabled)
