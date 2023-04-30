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

from panda3d.core import Vec2, Vec3, Vec4
from panda3d.core import TextNode as TextNodeImpl

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.rpobject import RPObject
from metadrive.render_pipeline.rpcore.loader import RPLoader


class TextNode(RPObject):
    """ Interface for the Panda3D TextNode. """
    def __init__(
        self,
        font="/$$rp/data/font/Roboto-Bold.ttf",
        pixel_size=16,
        align="left",
        pos=Vec2(0),
        color=Vec3(1),
        parent=None
    ):
        """ Constructs a new text node, forwaring the parameters to the internal
        panda3d implementation """
        RPObject.__init__(self)
        self._node = TextNodeImpl('FTN')
        self._node.set_text("")
        self._node.set_align(getattr(TextNodeImpl, "A_" + align))
        self._node.set_text_color(color.x, color.y, color.z, 1)

        if parent is None:
            parent = Globals.base.aspect2d

        self._nodepath = parent.attach_new_node(self._node)
        self._nodepath.set_pos(pos.x, 0, pos.y)

        font = RPLoader.load_font(font)
        # font.set_outline(Vec4(0, 0, 0, 0.78), 1.6, 0.37)
        font.set_outline(Vec4(0, 0, 0, 1), 1.6, 0.37)
        font.set_scale_factor(1.0)
        font.set_texture_margin(int(pixel_size / 4.0 * 2.0))
        font.set_bg(Vec4(0, 0, 0, 0))
        self._node.set_font(font)
        self.set_pixel_size(pixel_size)

    @property
    def text(self):
        """ Returns the current text """
        return self._node.get_text()

    @text.setter
    def text(self, text):
        """ Sets the current text """
        self._node.set_text(text)

    @property
    def color(self):
        """ Returns the current text color """
        return self._node.get_text_color()

    @color.setter
    def color(self, val):
        """ Sets the current text color """
        self._node.set_text_color(val)

    def set_pixel_size(self, size):
        """ Sets the text size in pixels """
        self._nodepath.set_scale(size * 2.0 / float(Globals.native_resolution.y))
