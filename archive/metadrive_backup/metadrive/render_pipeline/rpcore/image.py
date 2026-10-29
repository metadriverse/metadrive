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

from panda3d.core import Texture, GeomEnums

from metadrive.render_pipeline.rpcore.rpobject import RPObject
from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.render_target import RenderTarget


class ImageFormatTypes(object):  # pylint: disable=too-few-public-methods
    """ This is a small helper class to prevent pylint errors about the Image
    class not defining the enums. This class just copies the enum properties
    from panda3d's builtin Texture class. """

    T_float = Texture.T_float
    T_unsigned_byte = Texture.T_unsigned_byte
    T_int = Texture.T_int
    T_unsigned_short = Texture.T_unsigned_short
    T_unsigned_int_24_8 = Texture.T_unsigned_int_24_8

    TT_buffer_texture = Texture.TT_buffer_texture
    TT_2d_texture = Texture.TT_2d_texture
    TT_3d_texture = Texture.TT_3d_texture
    TT_cube_map = Texture.TT_cube_map
    TT_cube_map_array = Texture.TT_cube_map_array
    TT_2d_texture_array = Texture.TT_2d_texture_array

    format_format = Texture.format_format
    format_component_type = Texture.format_component_type


class Image(RPObject, Texture, ImageFormatTypes):
    """ This is a wrapper arround the Texture class from Panda3D, which keeps
    track of all images and registers / unregisters them aswell as counting
    the memory used. This is used by all classes instead of pandas builtin
    Texture class. """

    # All registered images
    REGISTERED_IMAGES = []

    # String formats
    FORMAT_MAPPINGS = {
        "R11G11B10": (Texture.T_float, Texture.F_r11_g11_b10),
        "RGBA8": (Texture.T_unsigned_byte, Texture.F_rgba8),
        "RGBA16": (Texture.T_float, Texture.F_rgba16),
        "RGBA32": (Texture.T_float, Texture.F_rgba32),
        "R8": (Texture.T_unsigned_byte, Texture.F_red),
        "R8UI": (Texture.T_unsigned_byte, Texture.F_red),
        "R16": (Texture.T_float, Texture.F_r16),
        "R16UI": (Texture.T_unsigned_short, Texture.F_r16i),
        "R32": (Texture.T_float, Texture.F_r32),
        "R32I": (Texture.T_int, Texture.F_r32i),
    }

    @classmethod
    def create_buffer(cls, name, size, component_format):
        """ Creates a new buffer texture """
        img = cls("ImgBuffer-" + name)
        comp_type, comp_format = cls.convert_texture_format(component_format)
        img.setup_buffer_texture(size, comp_type, comp_format, GeomEnums.UH_static)
        return img

    @classmethod
    def create_counter(cls, name):
        """ Creates a new 1x1 R32I texture to be used as an atomic counter """
        return cls.create_buffer(name, 1, "R32I")

    @classmethod
    def create_2d(cls, name, w, h, component_format):
        """ Creates a new 2D texture """
        img = cls("Img2D-" + name)
        comp_type, comp_format = cls.convert_texture_format(component_format)
        img.setup_2d_texture(w, h, comp_type, comp_format)
        return img

    @classmethod
    def create_2d_array(cls, name, w, h, slices, component_format):
        """ Creates a new 2D-array texture """
        img = cls("Img2DArr-" + name)
        comp_type, comp_format = cls.convert_texture_format(component_format)
        img.setup_2d_texture_array(w, h, slices, comp_type, comp_format)
        return img

    @classmethod
    def create_3d(cls, name, w, h, slices, component_format):
        """ Creates a new 3D texture """
        img = cls("Img3D-" + name)
        comp_type, comp_format = cls.convert_texture_format(component_format)
        img.setup_3d_texture(w, h, slices, comp_type, comp_format)
        return img

    @classmethod
    def create_cube(cls, name, size, component_format):
        """ Creates a new cubemap """
        img = cls("ImgCube-" + name)
        comp_type, comp_format = cls.convert_texture_format(component_format)
        img.setup_cube_map(size, comp_type, comp_format)
        return img

    @classmethod
    def create_cube_array(cls, name, size, num_cubemaps, component_format):
        """ Creates a new cubemap """
        img = cls("ImgCubeArr-" + name)
        comp_type, comp_format = cls.convert_texture_format(component_format)
        img.setup_cube_map_array(size, num_cubemaps, comp_type, comp_format)
        return img

    @classmethod
    def convert_texture_format(cls, comp_type):
        """ Converts a string like 'RGBA8' to a texture type and format """
        return cls.FORMAT_MAPPINGS[comp_type]

    def __init__(self, name):
        """ Internal method to create a new image """
        RPObject.__init__(self, name)
        Texture.__init__(self, name)
        Image.REGISTERED_IMAGES.append(self)
        self.set_clear_color(0)
        self.clear_image()
        self.sort = RenderTarget.CURRENT_SORT

    def __del__(self):
        """ Destroys the image """
        self.warn("Image destructor not implemented yet")

    def write(self, pth):
        """ Writes the image to disk """
        Globals.base.graphicsEngine.extract_texture_data(self, Globals.base.win.gsg)
        if self.get_texture_type() in [Texture.TT_3d_texture, Texture.TT_cube_map]:
            Texture.write(self, "#_" + pth, 0, 0, True, False)
        else:
            Texture.write(self, pth)
