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

from panda3d.core import Vec2, TransparencyAttrib, Vec3
from direct.gui.DirectFrame import DirectFrame
from direct.gui.DirectButton import DirectButton
from direct.gui.DirectGui import DGG

from metadrive.render_pipeline.rpcore.rpobject import RPObject
from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.gui.text import Text


class DraggableWindow(RPObject):
    """ This is a simple draggable but not resizeable window """
    def __init__(self, width=800, height=500, title="Window", parent=None):
        """ Constructs a new window with the given dimensions and title """
        RPObject.__init__(self, "Window-" + title)
        self._width = width
        self._height = height
        self._title = title
        self._visible = True
        self._parent = parent if parent else Globals.base.pixel2d
        self._dragging = False
        self._drag_offset = Vec2(0)
        self._pos = Vec2(0)

    def center_on_screen(self):
        """ Centers the window on screen """
        self._context_scale = 1.0 / self._parent.get_sx()
        self._context_width = Globals.native_resolution.x * self._context_scale
        self._context_height = Globals.native_resolution.y * self._context_scale
        self._set_pos(Vec2((self._context_width - self._width) / 2, (self._context_height - self._height) / 2))

    def set_title(self, title):
        """ Sets the window title """
        self._title = title
        self._window_title.set_text(title)

    def show(self):
        """ Shows the window """
        self._visible = True
        self.center_on_screen()
        self._node.show()

    def hide(self):
        """ Hides the window """
        self._visible = False
        self._stop_drag()
        self._node.hide()

    def remove(self):
        """ Removes the window from the scene graph. You should still delete the
        instance """
        self._stop_drag()
        self._node.remove_node()

    def _create_components(self):
        """ Creates the window components """
        self._node = self._parent.attach_new_node("Window")
        self._node.set_pos(self._pos.x, 1, -self._pos.y)
        border_px = 1
        border_frame_size = (-border_px, self._width + border_px, border_px, -self._height - border_px)
        self._border_frame = DirectFrame(
            pos=(0, 1, 0),
            frameSize=border_frame_size,
            frameColor=(24 / 255.0, 131 / 255.0, 215 / 255.0, 1),
            parent=self._node,
            state=DGG.NORMAL
        )
        self._background = DirectFrame(
            pos=(0, 1, 0),
            frameSize=(0, self._width, 0, -self._height),
            frameColor=(0.1, 0.1, 0.1, 1.0),
            parent=self._node
        )
        self._title_bar = DirectFrame(
            pos=(0, 1, 0),
            frameSize=(0, self._width, 0, -25),
            # frameColor=(0.058, 0.058, 0.058, 1),
            frameColor=(1, 1, 1, 1),
            parent=self._node,
            state=DGG.NORMAL
        )
        self._window_title = Text(
            parent=self._node, x=8, y=17, text=self._title, size=13, color=Vec3(0.15), may_change=True
        )
        self._btn_close = DirectButton(
            relief=DGG.FLAT,
            pressEffect=1,
            pos=(self._width - 22, 1, -12),
            frameColor=(1.0, 0.2, 0.2, 0.5),
            parent=self._node,
            scale=(45 / 2, 1, 24 / 2),
            image="/$$rp/data/gui/close_window.png"
        )

        # Init bindings
        self._btn_close.set_transparency(TransparencyAttrib.M_alpha)
        self._btn_close.bind(DGG.B1CLICK, self._request_close)
        self._btn_close.bind(DGG.WITHIN, self._on_close_btn_hover)
        self._btn_close.bind(DGG.WITHOUT, self._on_close_btn_out)
        self._title_bar.bind(DGG.B1PRESS, self._start_drag)
        self._title_bar.bind(DGG.B1RELEASE, self._stop_drag)

    def _start_drag(self, evt=None):  # pylint: disable=unused-argument
        """ Gets called when the user starts dragging the window """
        self._dragging = True
        self._node.detach_node()
        self._node.reparent_to(self._parent)
        Globals.base.taskMgr.add(self._on_tick, "UIWindowDrag", uponDeath=self._stop_drag)
        self._drag_offset = self._pos - self._get_mouse_pos()

    def _on_close_btn_hover(self, evt=None):  # pylint: disable=unused-argument
        """ Internal method when the close button got hovered """
        self._btn_close["image"] = "/$$rp/data/gui/close_window_hover.png"

    def _on_close_btn_out(self, evt=None):  # pylint: disable=unused-argument
        """ Internal method when the close button is no longer hovered """
        self._btn_close["image"] = "/$$rp/data/gui/close_window.png"

    def _request_close(self, evt=None):  # pylint: disable=unused-argument
        """ This method gets called when the close button gets clicked """
        self.hide()

    def _stop_drag(self, evt=None):  # pylint: disable=unused-argument
        """ Gets called when the user stops dragging the window """
        Globals.base.taskMgr.remove("UIWindowDrag")
        self._dragging = False

    def _get_mouse_pos(self):
        """ Internal helper function to get the mouse position, scaled by
        the context scale """
        mouse_x, mouse_y = (Globals.base.win.get_pointer(0).x, Globals.base.win.get_pointer(0).y)
        return Vec2(mouse_x, mouse_y) * self._context_scale

    def _set_pos(self, pos):
        """ Moves the window to the specified position """
        self._pos = pos
        self._pos.x = max(self._pos.x, -self._width + 100)
        self._pos.y = max(self._pos.y, 25)
        self._pos.x = min(self._pos.x, self._context_width - 100)
        self._pos.y = min(self._pos.y, self._context_height - 50)
        self._node.set_pos(self._pos.x, 1, -self._pos.y)

    def _on_tick(self, task):
        """ Task which updates the window while being dragged """
        self._set_pos(self._get_mouse_pos() + self._drag_offset)
        return task.cont
