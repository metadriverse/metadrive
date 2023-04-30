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
from functools import partial

from panda3d.core import Vec3

from metadrive.render_pipeline.rplibs.yaml import load_yaml_file

from metadrive.render_pipeline.rpcore.native import NATIVE_CXX_LOADED
from metadrive.render_pipeline.rpcore.gui.draggable_window import DraggableWindow
from metadrive.render_pipeline.rpcore.gui.labeled_checkbox import LabeledCheckbox
from metadrive.render_pipeline.rpcore.gui.checkbox_collection import CheckboxCollection


class RenderModeSelector(DraggableWindow):
    """ Window which offers the user to select a render mode to apply """
    def __init__(self, pipeline, parent):
        DraggableWindow.__init__(self, width=690, height=340, parent=parent, title="Select render mode")
        self._pipeline = pipeline
        self._selected_mode = ""
        self._create_components()
        self.hide()

    def _create_components(self):
        """ Internal method to init the components """
        DraggableWindow._create_components(self)
        self._content_node = self._node.attach_new_node("content")
        self._populate_content()

    def _populate_content(self):
        """ Populates the windows content """
        self._content_node.node().remove_all_children()

        # Reload config each time the window is opened so its easy to add new
        # render modes
        config = load_yaml_file("/$$rpconfig/debugging.yaml")

        debugger_content = self._content_node.attach_new_node("RenderModes")
        debugger_content.set_z(-20)
        debugger_content.set_x(20)

        render_modes = [("Default", "", False, "", False)]

        # Read modes from configuration
        for mode in config["render_modes"]:
            data = [mode["name"], mode["key"]]
            data.append(mode.get("cxx_only", False))
            data.append(mode.get("requires", ""))
            data.append(mode.get("special", False))
            render_modes.append(data)

        collection = CheckboxCollection()

        max_column_height = 9

        for idx, (mode, mode_id, requires_cxx, requires_plugin, special) in enumerate(render_modes):
            offs_y = (idx % max_column_height) * 24 + 35
            offs_x = (idx // max_column_height) * 220
            enabled = True
            if requires_cxx and not NATIVE_CXX_LOADED:
                enabled = False

            if requires_plugin:
                if not self._pipeline.plugin_mgr.is_plugin_enabled(requires_plugin):
                    enabled = False

            box = LabeledCheckbox(
                parent=debugger_content,
                x=offs_x,
                y=offs_y,
                text=mode.upper(),
                text_color=Vec3(0.4),
                radio=True,
                chb_checked=(mode_id == self._selected_mode),
                chb_callback=partial(self._set_render_mode, mode_id, special),
                text_size=14,
                expand_width=230,
                enabled=enabled
            )
            collection.add(box.checkbox)

    def _set_render_mode(self, mode_id, special, value):
        """ Callback which gets called when a render mode got selected """
        if not value:
            return

        to_remove = []
        for define in self._pipeline.stage_mgr.defines:
            if define.startswith("_RM_"):
                to_remove.append(define)
        for define in to_remove:
            del self._pipeline.stage_mgr.defines[define]

        if mode_id == "":
            self._pipeline.stage_mgr.defines["ANY_DEBUG_MODE"] = 0
        else:
            # Don't activate the generic debugging mode for special modes. This
            # is for modes like luminance which expect the scene to be rendered
            # unaltered.
            self._pipeline.stage_mgr.defines["ANY_DEBUG_MODE"] = 0 if special else 1
            self._pipeline.stage_mgr.defines["_RM_" + mode_id] = 1

        self._selected_mode = mode_id
        self._pipeline.reload_shaders()

    def toggle(self):
        """ Toggles the visibility of this windows """
        if self._visible:
            self.hide()
        else:
            self._populate_content()
            self.show()
