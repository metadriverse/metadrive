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

from panda3d.core import TransparencyAttrib, SamplerState
from direct.gui.DirectCheckBox import DirectCheckBox
import direct.gui.DirectGuiGlobals as DGG

from metadrive.render_pipeline.rpcore.rpobject import RPObject
from metadrive.render_pipeline.rpcore.loader import RPLoader


class Checkbox(RPObject):
    """ This is a wrapper around DirectCheckBox, providing a simpler interface
    and better visuals """
    def __init__(
        self,
        parent=None,
        x=0,
        y=0,
        callback=None,
        extra_args=None,
        radio=False,
        expand_width=100,
        checked=False,
        enabled=True
    ):
        RPObject.__init__(self)

        prefix = "checkbox" if not radio else "radiobox"

        if enabled:
            checked_img = RPLoader.load_texture("/$$rp/data/gui/" + prefix + "_checked.png")
            unchecked_img = RPLoader.load_texture("/$$rp/data/gui/" + prefix + "_default.png")
        else:
            checked_img = RPLoader.load_texture("/$$rp/data/gui/" + prefix + "_disabled.png")
            unchecked_img = checked_img

        # Set near filter, otherwise textures look like crap
        for tex in [checked_img, unchecked_img]:
            tex.set_minfilter(SamplerState.FT_linear)
            tex.set_magfilter(SamplerState.FT_linear)
            tex.set_wrap_u(SamplerState.WM_clamp)
            tex.set_wrap_v(SamplerState.WM_clamp)
            tex.set_anisotropic_degree(0)

        self._node = DirectCheckBox(
            parent=parent,
            pos=(x + 11, 1, -y - 8),
            scale=(10 / 2.0, 1, 10 / 2.0),
            checkedImage=checked_img,
            uncheckedImage=unchecked_img,
            image=unchecked_img,
            extraArgs=extra_args,
            state=DGG.NORMAL,
            relief=DGG.FLAT,
            command=self._update_status
        )

        self._node["frameColor"] = (0, 0, 0, 0)
        self._node["frameSize"] = (-2.6, 2 + expand_width / 7.5, -2.35, 2.5)
        self._node.set_transparency(TransparencyAttrib.M_alpha)

        self._callback = callback
        self._extra_args = extra_args
        self._collection = None

        if checked:
            self.set_checked(True, False)

    @property
    def collection(self):
        """ Returns a handle to the assigned checkbox collection, or None
        if no collection was assigned """
        return self._collection

    @collection.setter
    def collection(self, coll):
        """ Internal method to add a checkbox to a checkbox collection, this
        is used for radio-buttons """
        self._collection = coll

    @property
    def checked(self):
        """ Returns whether the node is currently checked """
        return self._node["isChecked"]

    @property
    def node(self):
        """ Returns a handle to the internally used node """
        return self._node

    def _update_status(self, status):
        """ Internal method when another checkbox in the same radio group
        changed it's value """

        if not status and self._collection:
            self._node.commandFunc(None)
            return

        if self._collection:
            if status:
                self._collection.on_checkbox_changed(self)
                # A radio box can't be unchecked
                # self._node["state"] = DGG.DISABLED

        if self._callback is not None:
            self._callback(*([status] + self._extra_args))

    def set_checked(self, val, do_callback=True):
        """ Internal method to check/uncheck the checkbox """
        self._node["isChecked"] = val

        if val:
            self._node["image"] = self._node["checkedImage"]
        else:
            self._node["image"] = self._node["uncheckedImage"]

        if do_callback and self._callback is not None:
            self._callback(*([val] + self._extra_args))
