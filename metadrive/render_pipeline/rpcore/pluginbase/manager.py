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

import importlib
import collections

from metadrive.render_pipeline.rplibs.six import iteritems, itervalues
from metadrive.render_pipeline.rplibs.yaml import load_yaml_file

from direct.stdpy.file import listdir, isdir, join, open

from metadrive.render_pipeline.rpcore.rpobject import RPObject
from metadrive.render_pipeline.rpcore.native import NATIVE_CXX_LOADED
from metadrive.render_pipeline.rpcore.pluginbase.setting_types import make_setting_from_data
from metadrive.render_pipeline.rpcore.pluginbase.day_setting_types import make_daysetting_from_data


class PluginManager(RPObject):
    """ This class manages all plugins. It provides functionality to load plugin
    settings, trigger callbacks on plugins, initialize the plugin instances
    and much more. """
    def __init__(self, pipeline):
        """ Constructs a new manager with no plugins loaded. To load settings
        and plugins, call load(). """
        RPObject.__init__(self)
        self._pipeline = pipeline
        self.settings = {}
        self.day_settings = {}
        self.instances = {}
        self.enabled_plugins = set()

        # Used by the plugin configurator and to only load the required data
        self.requires_daytime_settings = True

    def load(self):
        """ Loads all plugins and their settings, and also constructs instances
        of the main plugin classes for all enabled plugins """
        self.debug("Loading plugin settings")
        self.load_base_settings("/$$rp/rpplugins")
        self.load_setting_overrides("/$$rpconfig/plugins.yaml")

        if self.requires_daytime_settings:
            self.load_daytime_overrides("/$$rpconfig/daytime.yaml")

        self.debug("Creating plugin instances ..")
        for plugin_id in self.settings:
            handle = self._load_plugin(plugin_id)
            if handle:
                self.instances[plugin_id] = handle
            else:
                self.disable_plugin(plugin_id)

    def disable_plugin(self, plugin_id):
        """ Disables a plugin, given its plugin_id. This will remove it from
        the list of enabled plugins, if it ever was there """
        self.warn("Disabling", plugin_id)
        if plugin_id in self.enabled_plugins:
            self.enabled_plugins.remove(plugin_id)
        for instance in list(self.instances.values()):
            if plugin_id in instance.required_plugins:
                self.disable_plugin(instance.plugin_id)
        # if plugin_id in self.instances:
        #     del self.instances[plugin_id]

    def unload(self):
        """ Unloads all plugins """
        self.debug("Unloading all plugins")
        self.instances = {}
        self.settings = {}
        self.day_settings = {}
        self.enabled_plugins = set()

    def update(self):
        """ Main update method """
        pass

    def load_base_settings(self, plugin_dir):
        """ Loads the base settings of all plugins, even of disabled plugins.
        This is required to verify all overrides. """
        for entry in listdir(plugin_dir):
            abspath = join(plugin_dir, entry)
            if isdir(abspath) and entry not in ("__pycache__", "plugin_prefab"):
                self.load_plugin_settings(entry, abspath)

    def load_plugin_settings(self, plugin_id, plugin_pth):
        """ Internal method to load all settings of a plugin, given its plugin
        id and path to the plugin base directory """
        config_file = join(plugin_pth, "config.yaml")
        config = load_yaml_file(config_file)
        # When you don't specify anything in the settings, instead of
        # returning an empty dictionary, pyyaml returns None
        config["settings"] = config["settings"] or []
        config["daytime_settings"] = config["daytime_settings"] or []

        if config["settings"] and len(config["settings"][0]) != 2:
            self.fatal("Invalid plugin configuration, did you miss '!!omap'?")

        settings = collections.OrderedDict([(k, make_setting_from_data(v)) for k, v in config["settings"]])
        self.settings[plugin_id] = settings

        if self.requires_daytime_settings:
            daysettings = collections.OrderedDict(
                [(k, make_daysetting_from_data(v)) for k, v in config["daytime_settings"]]
            )
            self.day_settings[plugin_id] = daysettings

    def load_setting_overrides(self, override_path):
        """ Loads an override file for the settings, which contains values to
        override the settings with """
        overrides = load_yaml_file(override_path)
        if not overrides:
            self.warn("Failed to load overrides")
            return
        self.enabled_plugins = set(overrides["enabled"] or [])
        for plugin_id, pluginsettings in iteritems(overrides["overrides"] or {}):
            if plugin_id not in self.settings:
                self.warn("Unkown plugin in plugin config:", plugin_id)
                continue
            for setting_id, setting_val in iteritems(pluginsettings or {}):
                if setting_id not in self.settings[plugin_id]:
                    self.warn("Unkown override:", plugin_id, ":", setting_id)
                    continue
                self.settings[plugin_id][setting_id].set_value(setting_val)

    def load_daytime_overrides(self, override_path):
        """ Loads an override file for the daytime settings, which contains
        values to override the settings with """
        overrides = load_yaml_file(override_path)
        if not overrides:
            self.warn("Failed to load daytime overrides")
            return
        for plugin_id, settings in iteritems(overrides["control_points"] or {}):
            for setting_id, control_points in iteritems(settings):
                if setting_id not in self.day_settings[plugin_id]:
                    self.warn("Unkown daytime override:", plugin_id, ":", setting_id)
                    continue
                self.day_settings[plugin_id][setting_id].set_control_points(control_points)

    def trigger_hook(self, hook_name):
        """ Triggers a given hook on all plugins, effectively calling all
        bound callbacks """
        hook_method = "on_{}".format(hook_name)
        for plugin_id in self.enabled_plugins:
            plugin_handle = self.instances[plugin_id]
            if hasattr(plugin_handle, hook_method):
                getattr(plugin_handle, hook_method)()

    def is_plugin_enabled(self, plugin_id):
        """ Returns whether a plugin is currently enabled and loaded """
        return plugin_id in self.enabled_plugins

    def get_setting_handle(self, plugin_id, setting_id):
        """ Returns the handle to a setting """
        return self.settings[plugin_id][setting_id]

    def init_defines(self):
        """ Initializes all plugin settings as a define, so they can be queried
        in a shader """
        for plugin_id in self.enabled_plugins:
            pluginsettings = self.settings[plugin_id]
            self._pipeline.stage_mgr.defines["HAVE_PLUGIN_{}".format(plugin_id)] = 1
            for setting_id, setting in iteritems(pluginsettings):
                if setting.shader_runtime or not setting.runtime:
                    # Only store settings which either never change, or trigger
                    # a shader reload when they change
                    setting.add_defines(plugin_id, setting_id, self._pipeline.stage_mgr.defines)

    def _load_plugin(self, plugin_id):
        """ Internal method to load a plugin """
        plugin_class = "metadrive.render_pipeline.rpplugins.{}.plugin".format(plugin_id)
        module = importlib.import_module(plugin_class)
        instance = module.Plugin(self._pipeline)
        if instance.native_only and not NATIVE_CXX_LOADED:
            if plugin_id in self.enabled_plugins:
                self.warn("Cannot load", plugin_id, "since it requires the C++ modules.")
                return False
        for required_plugin in instance.required_plugins:
            if required_plugin not in self.enabled_plugins:
                if plugin_id in self.enabled_plugins:
                    self.warn("Cannot load {} since it requires {}".format(plugin_id, required_plugin))
                    return False
                break
        return instance

    def save_overrides(self, override_file):
        """ Saves all overrides to the given file """
        output = "\n# Render Pipeline Plugin Configuration\n"
        output += "# Instead of editing this file, prefer to use the Plugin Configurator\n"
        output += "# Any formatting and comments will be lost\n\n"
        output += "enabled:\n"

        def sort_criteria(pid):
            return ("A" if self.is_plugin_enabled(pid) else "B") + pid

        for plugin_id in sorted(self.settings, key=sort_criteria):
            output += "   {}- {}\n".format(" # " if plugin_id not in self.enabled_plugins else " ", plugin_id)
        output += "\n\noverrides:\n"
        for plugin_id, pluginsettings in sorted(iteritems(self.settings)):
            output += " " * 4 + plugin_id + ":\n"
            for setting_id, setting_handle in iteritems(pluginsettings):
                output += " " * 8 + "{}: {}\n".format(setting_id, setting_handle.value)
            output += "\n"
        with open(override_file, "w") as handle:
            handle.write(output)

    def save_daytime_overrides(self, override_file):
        """ Saves all time of day overrides to the given file """
        output = "\n# Render Pipeline Time Of Day Configuration\n"
        output += "# Instead of editing this file, prefer to use the Time Of Day Editor\n"
        output += "# Any formatting and comments will be lost\n\n"
        output += "control_points:\n"
        for plugin_id, settings in iteritems(self.day_settings):
            if settings:
                output += " " * 4 + plugin_id + ":\n"
                for setting_id, setting_handle in iteritems(settings):
                    output += " " * 8 + setting_id + ": "
                    output += setting_handle.serialize() + "\n"
        with open(override_file, "w") as handle:
            handle.write(output)

    def set_plugin_enabled(self, plugin_id, enabled=True):
        """ Sets whether a plugin is enabled or not, thus should be loaded when
        the pipeline starts or not """
        if enabled:
            self.enabled_plugins.add(plugin_id)
        else:
            self.enabled_plugins.remove(plugin_id)

    def reset_plugin_settings(self, plugin_id):
        """ Resets all settings of a given plugin """
        for setting in itervalues(self.settings[plugin_id]):
            setting.value = setting.default

    def on_setting_changed(self, plugin_id, setting_id, value):
        """ Callback when a setting got changed. This will update the setting,
        and also call the callback for that setting, in case the plugin defined
        one. """
        if plugin_id not in self.settings or setting_id not in self.settings[plugin_id]:
            self.warn("Got invalid setting change:", plugin_id, "/", setting_id)
            return

        setting = self.settings[plugin_id][setting_id]
        setting.set_value(value)

        if plugin_id not in self.enabled_plugins:
            return

        if setting.runtime or setting.shader_runtime:
            update_method = self.instances[plugin_id], "update_" + setting_id
            if hasattr(*update_method):
                getattr(*update_method)()

        if setting.shader_runtime:
            self.init_defines()
            self._pipeline.stage_mgr.write_autoconfig()
            self.instances[plugin_id].reload_shaders()
