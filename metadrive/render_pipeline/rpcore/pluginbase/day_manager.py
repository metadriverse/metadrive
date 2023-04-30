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

from metadrive.render_pipeline.rplibs.six import iteritems
from direct.stdpy.file import open

from metadrive.render_pipeline.rpcore.rpobject import RPObject
from metadrive.render_pipeline.rpcore.util.shader_input_blocks import GroupedInputBlock
from metadrive.render_pipeline.rpcore.pluginbase.day_setting_types import ColorType


class DayTimeManager(RPObject):
    """ This manager handles all time of day settings, provides them as
    a input to all shaders, and stores which time it currently is """
    def __init__(self, pipeline):
        RPObject.__init__(self)
        self._pipeline = pipeline
        self._input_ubo = GroupedInputBlock("TimeOfDay")
        self._time = 0.5
        self._setting_handles = {}

    @property
    def time(self):
        """ Returns the current time of day as floating point number
        from 0 to 1, whereas 0 means 00:00 and 1 means 24:00 (=00:00) """
        return self._time

    @time.setter
    def time(self, day_time):
        """ Sets the current time of day as floating point number from
        0 to 1, whereas 0 means 00:00 and 1 means 24:00 (=00:00). Any
        number greater than 1 will be reduced to fit the 0 .. 1 range by
        doing time modulo 1.

        Alternatively a string in the format 'hh:mm' can be passed. """
        if isinstance(day_time, float):
            self._time = day_time % 1.0
        elif isinstance(day_time, str):
            parts = [int(i) for i in day_time.split(":")]
            self._time = (parts[0] * 60 + parts[1]) / (24 * 60)

    @property
    def formatted_time(self):
        """ Returns the current time as formatted string, e.g. 12:34 """
        total_minutes = int(self._time * 24 * 60)
        return "{:02d}:{:02d}".format(total_minutes // 60, total_minutes % 60)

    def load_settings(self):
        """ Loads all day time settings from the plugin manager and registers
        them to the used input buffer """
        for plugin_id, settings in iteritems(self._pipeline.plugin_mgr.day_settings):
            for setting, handle in iteritems(settings):
                setting_id = "{}.{}".format(plugin_id, setting)
                self._input_ubo.register_pta(setting_id, handle.glsl_type)
                self._setting_handles[setting_id] = handle
        self._pipeline.stage_mgr.input_blocks.append(self._input_ubo)

        # Generate UBO shader code
        shader_code = self._input_ubo.generate_shader_code()
        with open("/$$rptemp/$$daytime_config.inc.glsl", "w") as handle:
            handle.write(shader_code)

    def update(self):
        """ Internal update method which updates all day time settings """
        for setting_id, handle in iteritems(self._setting_handles):
            # XXX: Find a better interface for this. Without this fix, colors
            # are in the range 0 .. 255 in the shader.
            if isinstance(handle, ColorType):
                value = handle.get_value_at(self._time)
            else:
                value = handle.get_scaled_value_at(self._time)
            self._input_ubo.update_input(setting_id, value)
