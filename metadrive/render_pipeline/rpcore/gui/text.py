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

from panda3d.core import Vec2, Vec3, TextNode, Vec4
from direct.gui.OnscreenText import OnscreenText

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.rpobject import RPObject


class Text(RPObject):
    """ Simple wrapper around OnscreenText, providing a simpler interface """
    def __init__(self, text="", parent=None, x=0, y=0, size=10, align="left", color=None, may_change=False, font=None):
        """ Constructs a new text. The parameters are almost equal to the
        parameters of OnscreenText """
        RPObject.__init__(self)

        if color is None:
            color = Vec3(1)

        align_mode = TextNode.A_left

        if align == "center":
            align_mode = TextNode.A_center
        elif align == "right":
            align_mode = TextNode.A_right

        if font is None:
            font = Globals.font
            # Should always have a global font. Never use the default panda font!
            assert font

        self._initial_pos = Vec2(x, -y)
        self._node = OnscreenText(
            text=text,
            parent=parent,
            pos=self._initial_pos,
            scale=size,
            align=align_mode,
            fg=Vec4(color.x, color.y, color.z, 1.0),
            font=font,
            mayChange=may_change
        )

    @property
    def node(self):
        """ Returns a handle to the internlally used node """
        return self._node

    def set_text(self, text):
        """ Changes the text, remember to pass may_change to the constructor,
        otherwise this method does not work. """
        self._node["text"] = text

    def get_initial_pos(self):
        """ Returns the initial position of the text. This can be used for
        animations """
        return self._initial_pos

    def pos_interval(self, *args, **kwargs):
        """ Returns a pos interval, this is a wrapper around
        NodePath.pos_interval """
        return self._node.pos_interval(*args, **kwargs)
