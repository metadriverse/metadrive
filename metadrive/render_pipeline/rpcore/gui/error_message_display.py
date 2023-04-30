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

from __future__ import print_function

from panda3d.core import Notify, LineStream, Vec3

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.rpobject import RPObject

from metadrive.render_pipeline.rpcore.gui.text import Text


class ErrorMessageDisplay(RPObject):
    """ This is a gui element which listens to the panda output stream
    and shows errors """
    def __init__(self):
        RPObject.__init__(self)
        self._num_errors = 0
        self._error_node = Globals.base.pixel2d.attach_new_node("ErrorDisplay")
        self._error_node.set_z(-180)
        self._notify_stream = None

    def _init_notify(self):
        """ Internal method to init the stream to catch all notify messages """
        self._notify_stream = LineStream()
        Notify.ptr().set_ostream_ptr(self._notify_stream, False)

    def update(self):
        """ Updates the error display, fetching all new messages from the notify
        stream """

        if not self._notify_stream:
            self._init_notify()

        while self._notify_stream.is_text_available():
            line = self._notify_stream.get_line().strip()
            if "warning" in line:
                RPObject.global_warn("Panda3D", line)
                # self.add_warning(line)
            elif "error" in line:
                RPObject.global_error("Panda3D", line)
                self.add_error(line)
            else:
                RPObject.global_debug("Panda3D", line)

    def add_error(self, msg):
        """ Adds a new error message """
        self.add_text(msg, Vec3(1, 0.2, 0.2))

    def add_warning(self, msg):
        """ Adds a new warning message """
        self.add_text(msg, Vec3(1, 1, 0.2))

    def add_text(self, text, color):
        """ Internal method to add a new text to the output """
        Text(
            x=Globals.native_resolution.x - 30,
            y=self._num_errors * 23,
            align="right",
            text=text,
            size=12,
            parent=self._error_node,
            color=color
        )

        self._num_errors += 1

        if self._num_errors > 30:
            self.clear_messages()
            self.add_error("Error count exceeded. Cleared errors ..")

    def clear_messages(self):
        """ Clears all messages / removes them """
        self._error_node.node().remove_all_children()
        self._num_errors = 0

    def show(self):
        self._error_node.show()

    def hide(self):
        self._error_node.hide()
