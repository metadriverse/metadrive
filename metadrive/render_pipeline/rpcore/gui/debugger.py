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

import os
import sys
import subprocess

from metadrive.render_pipeline.rplibs.six.moves import range  # pylint: disable=import-error

from panda3d.core import Vec4, Vec3, Vec2, RenderState, TransformState
from panda3d.core import TexturePool, SceneGraphAnalyzer
from direct.interval.IntervalGlobal import Sequence

from metadrive.render_pipeline.rpcore.gui.sprite import Sprite
from metadrive.render_pipeline.rpcore.gui.buffer_viewer import BufferViewer
from metadrive.render_pipeline.rpcore.gui.pipe_viewer import PipeViewer
from metadrive.render_pipeline.rpcore.gui.render_mode_selector import RenderModeSelector

from metadrive.render_pipeline.rpcore.gui.text_node import TextNode
from metadrive.render_pipeline.rpcore.gui.error_message_display import ErrorMessageDisplay
from metadrive.render_pipeline.rpcore.gui.exposure_widget import ExposureWidget
from metadrive.render_pipeline.rpcore.gui.fps_chart import FPSChart
from metadrive.render_pipeline.rpcore.gui.pixel_inspector import PixelInspector

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.rpobject import RPObject

from metadrive.render_pipeline.rpcore.native import NATIVE_CXX_LOADED
from metadrive.render_pipeline.rpcore.render_target import RenderTarget
from metadrive.render_pipeline.rpcore.image import Image


class Debugger(RPObject):
    """ This class manages the onscreen control, and displays statistics. """
    def __init__(self, pipeline):
        RPObject.__init__(self)
        self.debug("Creating debugger")
        self.pipeline = pipeline
        self.analyzer = SceneGraphAnalyzer()

        self.fullscreen_node = Globals.base.pixel2d.attach_new_node("rp_debugger")
        self.create_components()
        self.init_keybindings()

        if self.advanced_info:
            Globals.base.doMethodLater(
                0.5, lambda task: self.collect_scene_data(), "RPDebugger_collectSceneData_initial"
            )

        Globals.base.doMethodLater(0.1, self.update_stats, "RPDebugger_updateStats")

    @property
    def advanced_info(self):
        return self.pipeline.settings["pipeline.advanced_debugging_info"]

    def create_components(self):
        """ Creates the gui components """

        self.debugger_width = 460
        self.debugger_visible = False
        self.debugger_interval = None

        self.create_stats()
        self.create_hints()

        self.pipeline_logo = Sprite(
            image="/$$rp/data/gui/pipeline_logo_text.png", x=30, y=30, parent=self.fullscreen_node
        )

        if self.advanced_info:
            self.exposure_node = self.fullscreen_node.attach_new_node("ExposureWidget")
            self.exposure_widget = ExposureWidget(self.pipeline, self.exposure_node)

        self.fps_node = self.fullscreen_node.attach_new_node("FPSChart")
        self.fps_node.set_pos(Vec3(21, 1, -108 - 40))
        self.fps_widget = FPSChart(self.pipeline, self.fps_node)

        self.pixel_widget = PixelInspector(self.pipeline)
        self.buffer_viewer = BufferViewer(self.pipeline, self.fullscreen_node)
        self.pipe_viewer = PipeViewer(self.pipeline, self.fullscreen_node)
        self.rm_selector = RenderModeSelector(self.pipeline, self.fullscreen_node)
        self.error_msg_handler = ErrorMessageDisplay()

        self.handle_window_resize()

    def update(self):
        """ Updates the gui """
        self.error_msg_handler.update()
        self.pixel_widget.update()

    def collect_scene_data(self, task=None):
        """ Analyzes the scene graph to provide useful information """
        self.analyzer.clear()
        for geom_node in Globals.base.render.find_all_matches("**/+GeomNode"):
            self.analyzer.add_node(geom_node.node())
        if task:
            return task.again

    def create_stats(self):
        """ Creates the stats overlay """
        self.overlay_node = Globals.base.aspect2d.attach_new_node("Overlay")
        self.debug_lines = []

        num_lines = 6 if self.advanced_info else 1
        for i in range(num_lines):
            self.debug_lines.append(
                TextNode(pos=Vec2(0, -i * 0.046), parent=self.overlay_node, align="right", color=Vec3(0.7, 1, 1))
            )
        self.debug_lines[0].color = Vec4(1, 1, 0, 1)

    def create_hints(self):
        """ Creates the hints like keybindings and when reloading shaders """
        self.hint_reloading = Sprite(image="/$$rp/data/gui/shader_reload_hint.png", parent=self.fullscreen_node)
        self.set_reload_hint_visible(False)

        self.python_warning = None
        if not NATIVE_CXX_LOADED:
            # Warning when using the python version
            self.python_warning = Sprite(image="/$$rp/data/gui/python_warning.png", parent=self.fullscreen_node)
            Sequence(
                self.python_warning.color_scale_interval(0.7, Vec4(0.3, 1, 1, 0.7), blendType="easeOut"),
                self.python_warning.color_scale_interval(0.7, Vec4(1, 1, 1, 1.0), blendType="easeOut"),
            ).loop()

        # Keybinding hints
        self.keybinding_instructions = Sprite(
            image="/$$rp/data/gui/keybindings.png", parent=self.fullscreen_node, any_filter=False
        )

    def set_reload_hint_visible(self, flag):
        """ Sets whether the shader reload hint is visible """
        if flag:
            self.hint_reloading.show()
        else:
            self.hint_reloading.hide()

    def handle_window_resize(self):
        """ Handles the window resize, repositions the GUI elements to fit on
        screen. """
        # When using small resolutions, scale the GUI so its still useable,
        # otherwise the sub-windows are bigger than the main window
        self.gui_scale = max(0.65, min(1.0, Globals.native_resolution.x / 1920.0))
        self.fullscreen_node.set_scale(self.gui_scale)

        if self.advanced_info:
            self.exposure_node.set_pos(
                Globals.native_resolution.x // self.gui_scale - 200, 1,
                -Globals.native_resolution.y // self.gui_scale + 120
            )
        self.hint_reloading.set_pos(float((Globals.native_resolution.x) // 2) / self.gui_scale - 465 // 2, 220)
        self.keybinding_instructions.set_pos(
            30,
            Globals.native_resolution.y // self.gui_scale - 510.0,
        )
        self.overlay_node.set_pos(Globals.base.get_aspect_ratio() - 0.07, 1, 1.0 - 0.07)
        if self.python_warning:
            self.python_warning.set_pos(
                (Globals.native_resolution.x // self.gui_scale - 1054) // 2,
                (Globals.native_resolution.y // self.gui_scale - 118 - 40)
            )

        for text in self.debug_lines:
            text.set_pixel_size(16 * max(0.8, self.gui_scale))

        self.buffer_viewer.center_on_screen()
        self.pipe_viewer.center_on_screen()
        self.rm_selector.center_on_screen()

    def init_keybindings(self):
        """ Inits the debugger keybindings """
        Globals.base.accept("v", self.buffer_viewer.toggle)
        # Globals.base.accept("c", self.pipe_viewer.toggle)
        Globals.base.accept("z", self.rm_selector.toggle)
        Globals.base.accept("f5", self.toggle_gui_visible)
        Globals.base.accept("f6", self.toggle_keybindings_visible)
        Globals.base.accept("5", self.pipeline.reload_shaders)
        Globals.base.accept("m", self.start_material_editor)

    def start_material_editor(self):
        """ Starts the material editor """
        self.debug("Starting material editor")
        pth = sys.executable
        editor = os.path.dirname(os.path.realpath(__file__))
        editor = os.path.join(editor, "..", "..", "toolkit", "material_editor", "main.py")
        subprocess.Popen([pth, editor], shell=True)

    def toggle_gui_visible(self):
        """ Shows / Hides the gui """

        if not self.fullscreen_node.is_hidden():
            self.fullscreen_node.hide()
            self.overlay_node.hide()
            self.error_msg_handler.hide()
        else:
            self.fullscreen_node.show()
            self.overlay_node.show()
            self.error_msg_handler.show()

    def toggle_keybindings_visible(self):
        """ Shows / Hides the FPS graph """
        if not self.keybinding_instructions.is_hidden():
            self.keybinding_instructions.hide()
        else:
            self.keybinding_instructions.show()

    def update_stats(self, task=None):
        """ Updates the stats overlay """
        clock = Globals.clock
        self.debug_lines[0].text = "{:3.0f} fps  |  {:3.1f} ms  |  {:3.1f} ms max".format(
            clock.get_average_frame_rate(), 1000.0 / max(0.001, clock.get_average_frame_rate()),
            clock.get_max_frame_duration() * 1000.0
        )

        if not self.advanced_info:
            return task.again if task else None

        text = "{:4d} states |  {:4d} transforms "
        text += "|  {:4d} cmds |  {:4d} lights |  {:4d} shadows "
        text += "|  {:5.1f}% atlas usage"
        self.debug_lines[1].text = text.format(
            RenderState.get_num_states(), TransformState.get_num_states(),
            self.pipeline.light_mgr.cmd_queue.num_processed_commands, self.pipeline.light_mgr.num_lights,
            self.pipeline.light_mgr.num_shadow_sources, self.pipeline.light_mgr.shadow_atlas_coverage
        )

        text = "Internal:  {:3.0f} MB VRAM |  {:5d} img |  {:5d} tex |  "
        text += "{:5d} fbos |  {:3d} plugins |  {:2d}  views  ({:2d} active)"
        tex_memory, tex_count = self.buffer_viewer.stage_information

        views, active_views = 0, 0
        for target in RenderTarget.REGISTERED_TARGETS:
            if not target.create_default_region:
                num_regions = target.internal_buffer.get_num_display_regions()
                for i, region in enumerate(target.internal_buffer.get_display_regions()):
                    # Skip overlay display region
                    if i == 0 and num_regions > 1:
                        continue
                    views += 1
                    active_views += 1 if target.active and region.active else 0

        self.debug_lines[2].text = text.format(
            tex_memory / (1024**2), len(Image.REGISTERED_IMAGES), tex_count, RenderTarget.NUM_ALLOCATED_BUFFERS,
            len(self.pipeline.plugin_mgr.enabled_plugins), views, active_views
        )

        text = "Scene:   {:4.0f} MB VRAM |  {:3d} tex |  {:4d} geoms "
        text += "|  {:4d} nodes |  {:7,.0f} vertices"
        scene_tex_size = 0
        for tex in TexturePool.find_all_textures():
            scene_tex_size += tex.estimate_texture_memory()

        self.debug_lines[3].text = text.format(
            scene_tex_size / (1024**2),
            len(TexturePool.find_all_textures()),
            self.analyzer.get_num_geoms(),
            self.analyzer.get_num_nodes(),
            self.analyzer.get_num_vertices(),
        )

        sun_vector = Vec3(0)
        if self.pipeline.plugin_mgr.is_plugin_enabled("scattering"):
            sun_vector = self.pipeline.plugin_mgr.instances["scattering"].sun_vector

        text = "Time:  {} ({:1.3f}) |  Sun  {:0.2f} {:0.2f} {:0.2f}"
        text += " |  X {:3.1f}  Y {:3.1f}  Z {:3.1f}"
        text += " |  {:2d} tasks |  scheduled: {:2d}"
        self.debug_lines[4].text = text.format(
            self.pipeline.daytime_mgr.formatted_time,
            self.pipeline.daytime_mgr.time,
            sun_vector.x,
            sun_vector.y,
            sun_vector.z,
            Globals.base.camera.get_x(Globals.base.render),
            Globals.base.camera.get_y(Globals.base.render),
            Globals.base.camera.get_z(Globals.base.render),
            self.pipeline.task_scheduler.num_tasks,
            self.pipeline.task_scheduler.num_scheduled_tasks,
        )

        text = "Scene shadows:  "
        if "pssm" in self.pipeline.plugin_mgr.enabled_plugins:
            focus = self.pipeline.plugin_mgr.instances["pssm"].scene_shadow_stage.last_focus
            if focus is not None:
                text += "{:3.1f} {:3.1f} {:3.1f} r {:3.1f}".format(focus[0].x, focus[0].y, focus[0].z, focus[1])
            else:
                text += "none"
        else:
            text += "inactive"

        text += "   |  HPR  ({:3.1f}, {:3.1f}, {:3.1f})  |   {:4d} x {:4d} pixels @ {:3.1f} %"
        text += "   |  {:3d} x {:3d} tiles"
        self.debug_lines[5].text = text.format(
            Globals.base.camera.get_h(Globals.base.render),
            Globals.base.camera.get_p(Globals.base.render),
            Globals.base.camera.get_r(Globals.base.render),
            Globals.native_resolution.x,
            Globals.native_resolution.y,
            self.pipeline.settings["pipeline.resolution_scale"] * 100.0,
            self.pipeline.light_mgr.num_tiles.x,
            self.pipeline.light_mgr.num_tiles.y,
        )
        if task:
            return task.again
