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

import direct.gui.DirectGuiGlobals as DGG
from direct.gui.DirectSlider import DirectSlider

from metadrive.render_pipeline.rpcore.rpobject import RPObject


class Slider(RPObject):
    """ This is a simple wrapper around DirectSlider, providing a simpler
    interface """
    def __init__(
        self,
        x=0,
        y=0,
        parent=None,
        size=100,
        min_value=0,
        max_value=100,
        value=50,
        page_size=1,
        callback=None,
        extra_args=None
    ):
        """ Inits the slider """
        RPObject.__init__(self)
        if extra_args is None:
            extra_args = []

        # Scale has to be 2.0, otherwise there will be an error.
        self._node = DirectSlider(
            pos=(size * 0.5 + x, 1, -y),
            parent=parent,
            range=(min_value, max_value),
            value=value,
            pageSize=page_size,
            scale=2.0,
            command=callback,
            extraArgs=extra_args,
            frameColor=(0.0, 0.0, 0.0, 1),
            frameSize=(-size * 0.25, size * 0.25, -5, 5),
            relief=DGG.FLAT,
            thumb_frameColor=(0.35, 0.53, 0.2, 1.0),
            thumb_relief=DGG.FLAT,
            thumb_frameSize=(-2.5, 2.5, -5.0, 5.0),
        )

    @property
    def value(self):
        """ Returns the currently assigned value """
        return self._node["value"]

    @value.setter
    def value(self, value):
        """ Sets the value of the slider """
        self._node["value"] = value

    @property
    def node(self):
        """ Returns a handle to the internally used node """
        return self._node
