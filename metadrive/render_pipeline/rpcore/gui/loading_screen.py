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

from metadrive.render_pipeline.rplibs.six.moves import range  # pylint: disable=import-error

from metadrive.render_pipeline.rpcore.gui.sprite import Sprite
from metadrive.render_pipeline.rpcore.rpobject import RPObject
from metadrive.render_pipeline.rpcore.globals import Globals


class LoadingScreen(RPObject):
    """ This is the default loading screen used by the pipeline. It provides
    the ability to display a simple image during loading. The image should be
    in the format 16:9 and not too small, to avoid being blurred out. """
    def __init__(self, pipeline, image_source="/$$rp/data/gui/loading_screen_bg.txo"):
        """ Inits the loading screen with a given image source. By default,
        this is the pipeline loading screen, but it can be overridden. """
        RPObject.__init__(self)
        self.pipeline = pipeline
        self.image_source = image_source

    def create(self):
        """ Creates the gui components """
        screen_w, screen_h = Globals.native_resolution.x, Globals.native_resolution.y
        self.fullscreen_node = Globals.base.pixel2dp.attach_new_node("LoadingScreen")
        self.fullscreen_node.set_bin("fixed", 10)
        self.fullscreen_node.set_depth_test(False)

        scale_w = screen_w / 1920.0
        scale_h = screen_h / 1080.0
        scale = max(scale_w, scale_h)

        self.fullscreen_bg = Sprite(
            image=self.image_source,
            x=(screen_w - 1920.0 * scale) // 2,
            y=(screen_h - 1080.0 * scale) // 2,
            w=int(1920 * scale),
            h=int(1080 * scale),
            parent=self.fullscreen_node,
            near_filter=False
        )

        for _ in range(2):
            Globals.base.graphicsEngine.render_frame()

    def remove(self):
        """ Removes the loading screen """
        self.fullscreen_bg.node["image"].get_texture().release_all()
        self.fullscreen_node.remove_node()
