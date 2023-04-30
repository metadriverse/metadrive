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

from panda3d.core import ComputeNode, Vec4, Vec3
from direct.gui.DirectFrame import DirectFrame

from metadrive.render_pipeline.rpcore.gui.sprite import Sprite
from metadrive.render_pipeline.rpcore.gui.text import Text

from metadrive.render_pipeline.rpcore.rpobject import RPObject
from metadrive.render_pipeline.rpcore.image import Image
from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.loader import RPLoader


class ExposureWidget(RPObject):
    """ Widget to show the current exposure """
    def __init__(self, pipeline, parent):
        """ Inits the widget """
        RPObject.__init__(self)
        self._pipeline = pipeline
        self._parent = parent
        self._node = self._parent.attach_new_node("ExposureWidgetNode")
        self._create_components()

    def _create_components(self):
        """ Internal method to init the widgets components """

        self._node.hide()

        # Create the texture where the gui component is rendered inside
        self._storage_tex = Image.create_2d("ExposureDisplay", 140, 20, "RGBA8")
        self._storage_tex.set_clear_color(Vec4(0.2, 0.6, 1.0, 1.0))
        self._storage_tex.clear_image()

        self._bg_frame = DirectFrame(
            parent=self._node, frameColor=(0.1, 0.1, 0.1, 1.0), frameSize=(200, 0, -10, -85), pos=(0, 0, 0)
        )

        self._display_img = Sprite(image=self._storage_tex, parent=self._node, w=140, h=20, x=20, y=50)

        self._display_txt = Text(
            text="Current Exposure".upper(), parent=self._node, x=160, y=40, size=13, color=Vec3(0.8), align="right"
        )

        # Create the shader which generates the visualization texture
        self._cshader_node = ComputeNode("ExposureWidget")
        self._cshader_node.add_dispatch(140 // 10, 20 // 4, 1)

        self._cshader_np = self._node.attach_new_node(self._cshader_node)

        # Defer the further loading
        Globals.base.taskMgr.doMethodLater(1.0, self._late_init, "ExposureLateInit")

    def _late_init(self, task):
        """ Gets called after the pipeline initialized, this extracts the
        exposure texture from the stage manager """
        stage_mgr = self._pipeline.stage_mgr

        if "Exposure" not in stage_mgr.pipes:
            self.debug("Disabling exposure widget, could not find the exposure data.")
            self._node.remove_node()
            return

        self._node.show()

        exposure_tex = stage_mgr.pipes["Exposure"]
        self._cshader = RPLoader.load_shader("/$$rp/shader/visualize_exposure.compute.glsl")
        self._cshader_np.set_shader(self._cshader)
        self._cshader_np.set_shader_inputs(DestTex=self._storage_tex, ExposureTex=exposure_tex)

        return task.done
