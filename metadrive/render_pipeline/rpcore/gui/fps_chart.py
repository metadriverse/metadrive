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

from panda3d.core import ComputeNode, Vec4, Vec3, PTAInt, PTAFloat

from metadrive.render_pipeline.rpcore.gui.sprite import Sprite
from metadrive.render_pipeline.rpcore.gui.text import Text

from metadrive.render_pipeline.rpcore.rpobject import RPObject
from metadrive.render_pipeline.rpcore.image import Image
from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.loader import RPLoader


class FPSChart(RPObject):
    """ Widget to show the FPS as a chart """
    def __init__(self, pipeline, parent):
        """ Inits the widget """
        RPObject.__init__(self)
        self._pipeline = pipeline
        self._parent = parent
        self._node = self._parent.attach_new_node("FPSChartNode")
        self._create_components()

    def _create_components(self):
        """ Internal method to init the widgets components """

        # Create the buffer which stores the last FPS values
        self._storage_buffer = Image.create_buffer("FPSValues", 250, "R16")
        self._storage_buffer.set_clear_color(Vec4(0))
        self._storage_buffer.clear_image()

        self._store_index = PTAInt.empty_array(1)
        self._store_index[0] = 0

        self._current_ftime = PTAFloat.empty_array(1)
        self._current_ftime[0] = 16.0

        self._chart_ms_max = PTAFloat.empty_array(1)
        self._chart_ms_max[0] = 40

        # Create the texture where the gui component is rendered inside
        self._display_tex = Image.create_2d("FPSChartRender", 250, 120, "RGBA8")
        self._display_tex.set_clear_color(Vec4(0))
        self._display_tex.clear_image()
        self._display_img = Sprite(image=self._display_tex, parent=self._node, w=250, h=120, x=10, y=10)

        # Defer the further loading
        Globals.base.taskMgr.doMethodLater(0.3, self._late_init, "FPSChartInit")

    def _late_init(self, task):
        """ Gets called after the pipeline was initialized """
        self._display_txt = Text(text="40 ms", parent=self._node, x=20, y=25, size=13, color=Vec3(1), may_change=True)
        self._display_txt_bottom = Text(
            text="0 ms", parent=self._node, x=20, y=120, size=13, color=Vec3(1), may_change=True
        )

        # Create the shader which generates the visualization texture
        self._cshader_node = ComputeNode("FPSChartUpdateChart")
        self._cshader_node.add_dispatch(250 // 10, 120 // 4, 1)
        self._cshader_np = self._node.attach_new_node(self._cshader_node)

        self._cshader = RPLoader.load_shader("/$$rp/shader/fps_chart.compute.glsl")
        self._cshader_np.set_shader(self._cshader)
        self._cshader_np.set_shader_inputs(
            DestTex=self._display_tex,
            FPSValues=self._storage_buffer,
            index=self._store_index,
            maxMs=self._chart_ms_max
        )

        self._update_shader_node = ComputeNode("FPSChartUpdateValues")
        self._update_shader_node.add_dispatch(1, 1, 1)
        self._update_shader_np = self._node.attach_new_node(self._update_shader_node)
        self._ushader = RPLoader.load_shader("/$$rp/shader/fps_chart_update.compute.glsl")
        self._update_shader_np.set_shader(self._ushader)
        self._update_shader_np.set_shader_inputs(
            DestTex=self._storage_buffer, index=self._store_index, currentData=self._current_ftime
        )

        Globals.base.addTask(self._update, "UpdateFPSChart", sort=-50)

        return task.done

    def _update(self, task):
        """ Updates the widget """
        self._store_index[0] = (self._store_index[0] + 1) % 250
        self._current_ftime[0] = Globals.clock.get_dt() * 1000.0

        avg_fps = Globals.clock.get_average_frame_rate()
        if avg_fps > 122:
            self._chart_ms_max[0] = 10.0
        elif avg_fps > 62:
            self._chart_ms_max[0] = 20.0
        elif avg_fps > 32:
            self._chart_ms_max[0] = 40.0
        elif avg_fps > 15:
            self._chart_ms_max[0] = 70.0
        else:
            self._chart_ms_max[0] = 200.0

        self._display_txt.set_text(str(int(self._chart_ms_max[0])) + " ms")

        return task.cont
