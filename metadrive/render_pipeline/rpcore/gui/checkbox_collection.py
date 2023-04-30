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

from metadrive.render_pipeline.rpcore.rpobject import RPObject

import direct.gui.DirectGuiGlobals as DGG


class CheckboxCollection(RPObject):
    """ This is a container for multiple Checkboxes, controlling that
    only one checkbox of this collection is checked at one time
    (like a radio-button) """
    def __init__(self):
        RPObject.__init__(self)
        self._items = []

    def add(self, chb):
        """ Adds a Checkbox to this collection """
        if chb.collection is not None:
            self.error("Can't add checkbox as it already belongs " "to another collection!")
            return
        chb.collection = self
        self._items.append(chb)

    def remove(self, chb):
        """ Removes a checkbox from this collection """
        if chb.collection is not self:
            self.error("Can't remove the checkbox from this collection as it is not " "attached to this collection!")
            return
        chb.collection = None
        self._items.remove(chb)

    def on_checkbox_changed(self, chb):
        """ Internal callback when a checkbox got changed """
        for item in self._items:
            if item is not chb:
                item.set_checked(False)
                item.node["state"] = DGG.NORMAL
