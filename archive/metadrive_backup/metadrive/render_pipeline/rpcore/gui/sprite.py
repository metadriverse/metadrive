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

from panda3d.core import TransparencyAttrib, Vec3, Texture, SamplerState
from direct.gui.OnscreenImage import OnscreenImage

from metadrive.render_pipeline.rpcore.rpobject import RPObject
from metadrive.render_pipeline.rpcore.loader import RPLoader


class Sprite(RPObject):
    """ Simple wrapper arround OnscreenImage, providing a simpler interface """
    def __init__(
        self, image=None, parent=None, x=0, y=0, w=None, h=None, transparent=True, near_filter=True, any_filter=True
    ):
        """ Creates a new image, taking (x,y) as topleft coordinates.

        When near_filter is set to true, a near filter will be set to the
        texture passed. This provides sharper images.

        When any_filter is set to false, the passed image won't be modified at
        all. This enables you to display existing textures, otherwise the
        texture would get a near filter in the 3D View, too. """

        RPObject.__init__(self)

        if not isinstance(image, Texture):
            if not isinstance(image, str):
                self.warn("Invalid argument to image parameter:", image)
                return
            image = RPLoader.load_texture(image)

            if w is None or h is None:
                w, h = image.get_x_size(), image.get_y_size()
        else:
            if w is None or h is None:
                w = 10
                h = 10

        self._width, self._height = w, h
        self._initial_pos = self._translate_pos(x, y)

        self.node = OnscreenImage(
            image=image, parent=parent, pos=self._initial_pos, scale=(self._width / 2.0, 1, self._height / 2.0)
        )

        if transparent:
            self.node.set_transparency(TransparencyAttrib.M_alpha)

        tex = self.node.get_texture()

        # Apply a near filter, but only if the parent has no scale, otherwise
        # it will look weird
        if near_filter and any_filter and parent.get_sx() == 1.0:
            tex.set_minfilter(SamplerState.FT_nearest)
            tex.set_magfilter(SamplerState.FT_nearest)

        if any_filter:
            tex.set_anisotropic_degree(8)
            tex.set_wrap_u(SamplerState.WM_clamp)
            tex.set_wrap_v(SamplerState.WM_clamp)

    def get_initial_pos(self):
        """ Returns the initial position of the image. This can be used for
        animations """
        return self._initial_pos

    def pos_interval(self, *args, **kwargs):
        """ Returns a pos interval, this is a wrapper around
        NodePath.posInterval """
        return self.node.posInterval(*args, **kwargs)

    def hpr_interval(self, *args, **kwargs):
        """ Returns a hpr interval, this is a wrapper around
        NodePath.hprInterval """
        return self.node.hprInterval(*args, **kwargs)

    def color_scale_interval(self, *args, **kwargs):
        """ Returns a color scale interval, this is a wrapper around
        NodePath.colorScaleInterval """
        return self.node.colorScaleInterval(*args, **kwargs)

    def set_image(self, img):
        """ Sets the current image """
        self.node.set_image(img)

    def get_width(self):
        """ Returns the width of the image in pixels """
        return self._width

    def get_height(self):
        """ Returns the height of the image in pixels """
        return self._height

    def set_pos(self, x, y):
        """ Sets the position """
        self.node.set_pos(self._translate_pos(x, y))

    def _translate_pos(self, x, y):
        """ Converts 2d coordinates to pandas coordinate system """
        return Vec3(x + self._width / 2.0, 1, -y - self._height / 2.0)

    def set_shader(self, shader):
        """ Sets a shader to be used for rendering the image """
        self.node.set_shader(shader)

    def set_shader_input(self, *args):
        """ Sets a shader input on the image """
        self.node.set_shader_input(*args)

    def set_shader_inputs(self, **kwargs):
        """ Sets multiple shader inputs on the image """
        self.node.set_shader_inputs(**kwargs)

    def remove(self):
        """ Removes the image """
        self.node.remove()

    def hide(self):
        """ Hides the image """
        self.node.hide()

    def show(self):
        """ Shows the image if it was previously hidden """
        self.node.show()

    def is_hidden(self):
        """ Returns whether the image is hidden """
        return self.node.is_hidden()
