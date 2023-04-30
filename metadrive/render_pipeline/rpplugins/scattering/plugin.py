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

import math

from panda3d.core import Vec3

from metadrive.render_pipeline.rpcore.pluginbase.base_plugin import BasePlugin

from .scattering_stage import ScatteringStage
from .scattering_envmap_stage import ScatteringEnvmapStage
from .godray_stage import GodrayStage


class Plugin(BasePlugin):
    name = "Atmospheric Scattering"
    author = "tobspr <tobias.springer1@gmail.com>"
    description = (
        "This plugin adds support for Atmospheric Scattering, and a "
        "single sun, based on the work from Eric Bruneton. It also "
        "adds support for atmospheric fog."
    )
    version = "1.2"

    def on_pipeline_created(self):
        self.scattering_model.load()
        self.scattering_model.compute()

    def on_stage_setup(self):
        self.display_stage = self.create_stage(ScatteringStage)
        self.envmap_stage = self.create_stage(ScatteringEnvmapStage)

        if self.get_setting("enable_godrays"):
            self.godray_stage = self.create_stage(GodrayStage)

        # Load scattering method
        method = self.get_setting("scattering_method")

        self.debug("Loading scattering method for '" + method + "'")

        if method == "eric_bruneton":
            from .scattering_methods import ScatteringMethodEricBruneton
            self.scattering_model = ScatteringMethodEricBruneton(self)
        elif method == "hosek_wilkie":
            from .scattering_methods import ScatteringMethodHosekWilkie
            self.scattering_model = ScatteringMethodHosekWilkie(self)  # noqa # pylint: disable=redefined-variable-type
        else:
            self.error("Unrecognized scattering method!")

    @property
    def sun_vector(self):
        """ Returns the sun vector """
        sun_altitude = self.get_daytime_setting("sun_altitude")
        sun_azimuth = self.get_daytime_setting("sun_azimuth")
        theta = (90 - sun_altitude) / 180.0 * math.pi
        phi = sun_azimuth / 180.0 * math.pi
        sun_vector = Vec3(math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta))
        return sun_vector

    def on_pre_render_update(self):
        self.envmap_stage.active = self._pipeline.task_scheduler.is_scheduled("scattering_update_envmap")

    def on_shader_reload(self):
        self.scattering_model.compute()
