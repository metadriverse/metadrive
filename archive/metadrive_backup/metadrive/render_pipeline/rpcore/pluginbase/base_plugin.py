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

from direct.stdpy.file import join

from metadrive.render_pipeline.rpcore.rpobject import RPObject


class BasePlugin(RPObject):
    """ This is the base class for all plugins. All plugin classes derive from
    this class. Additionally there are a lot of helpful functions provided,
    such as creating render stages. """

    # Plugins can set this to indicate they have some C++ code, so they cannot
    # be used with the python version.
    native_only = False

    # Plugins can set this to require other plugins
    required_plugins = tuple()

    def __init__(self, pipeline):
        """ Inits the plugin """
        self._pipeline = pipeline
        self._assigned_stages = []
        self.plugin_id = str(self.__class__.__module__).split(".")[-2]
        RPObject.__init__(self, "plugin:" + self.plugin_id)
        self._set_debug_color("magenta", "bright")

    @property
    def base_path(self):
        """ Returns the path to the root directory of the plugin """
        return "/$$rp/rpplugins/{}/".format(self.plugin_id)

    def get_resource(self, pth):
        """ Converts a local path from the plugins resource directory into
        an absolute path """
        return join(self.base_path, "resources", pth)

    def get_shader_resource(self, pth):
        """ Converts a local path from the plugins shader directory into
        an absolute path """
        return join(self.base_path, "shader", pth)

    def create_stage(self, stage_type):
        """ Shortcut to create a new render stage from a given class type. It
        also registers the stage to the stage manager, and links the stage
        to the current plugin instance. """
        stage_handle = stage_type(self._pipeline)
        self._pipeline.stage_mgr.add_stage(stage_handle)
        self._assigned_stages.append(stage_handle)
        return stage_handle

    def get_setting(self, setting_id, plugin_id=None):
        """ Returns the value of a setting given by its setting id. If plugin_id
        is set, returns the setting of the given plugin """
        return self._pipeline.plugin_mgr.settings[plugin_id or self.plugin_id][setting_id].value

    def get_daytime_setting(self, setting_id, plugin_id=None):
        """ Returns the value of a time of day setting given by its setting
        id. If plugin_id is set, returns the setting of a gien plugin """
        handle = self._pipeline.plugin_mgr.day_settings[plugin_id or self.plugin_id][setting_id]
        return handle.get_scaled_value_at(self._pipeline.daytime_mgr.time)

    def get_plugin_instance(self, plugin_id):
        """ Returns the instance of a different plugin, given by its id.
        This should be only used to access plugins which are also in the
        current plugins requirements """
        return self._pipeline.plugin_mgr.instances[plugin_id]

    def is_plugin_enabled(self, plugin_id):
        """ Returns whether a plugin is enabled and loaded, given is plugin id """
        return self._pipeline.plugin_mgr.is_plugin_enabled(plugin_id)

    def reload_shaders(self):
        """ Reloads all shaders of the plugin """
        for stage in self._assigned_stages:
            stage.reload_shaders()
