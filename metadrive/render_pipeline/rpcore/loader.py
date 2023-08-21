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

import time

from panda3d.core import PNMImage, VirtualFileSystem, VirtualFileMountRamdisk
from panda3d.core import Shader

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.rpobject import RPObject

__all__ = ("RPLoader", )


class timed_loading_operation(object):  # noqa # pylint: disable=invalid-name,too-few-public-methods
    """ Context manager for a synchronous loading operation, keeping track
    on how much time elapsed during the loading process, and warning about
    long loading times. """

    WARNING_COUNT = 0

    def __init__(self, resource):
        self.resource = resource
        if isinstance(self.resource, (list, tuple)):
            self.resource = ', '.join(self.resource)

    def __enter__(self):
        self.start_time = time.process_time()

    def __exit__(self, *args):
        duration = (time.process_time() - self.start_time) * 1000.0
        if duration > 80.0 and timed_loading_operation.WARNING_COUNT < 5:
            RPObject.global_debug("RPLoader", "Loading '" + self.resource + "' took", round(duration, 2), "ms")
            timed_loading_operation.WARNING_COUNT += 1
            if timed_loading_operation.WARNING_COUNT == 5:
                RPObject.global_debug("RPLoader", "Skipping further loading warnings (max warning count reached)")


class RPLoader(RPObject):
    """ Generic loader class used by the pipeline. All loading of assets happens
    here, which enables us to keep track of used resources """
    @classmethod
    def load_texture(cls, filename):
        """ Loads a 2D-texture from disk """
        with timed_loading_operation(filename):
            return Globals.base.loader.load_texture(filename)

    @classmethod
    def load_cube_map(cls, filename, read_mipmaps=False):
        """ Loads a cube map from disk """
        with timed_loading_operation(filename):
            return Globals.base.loader.load_cube_map(filename, readMipmaps=read_mipmaps)

    @classmethod
    def load_3d_texture(cls, filename):
        """ Loads a 3D-texture from disk """
        with timed_loading_operation(filename):
            return Globals.base.loader.load_3d_texture(filename)

    @classmethod
    def load_font(cls, filename):
        """ Loads a font from disk """
        with timed_loading_operation(filename):
            return Globals.base.loader.load_font(filename)

    @classmethod
    def load_shader(cls, *args):
        """ Loads a shader from disk """
        with timed_loading_operation(args):
            if len(args) == 1:
                return Shader.load_compute(Shader.SL_GLSL, args[0])
            return Shader.load(Shader.SL_GLSL, *args)

    @classmethod
    def load_model(cls, filename):
        """ Loads a model from disk """
        with timed_loading_operation(filename):
            return Globals.base.loader.load_model(filename)

    @classmethod
    def load_sliced_3d_texture(cls, fname, tile_size_x, tile_size_y=None, num_tiles=None):
        """ Loads a texture from the given filename and dimensions. If only
        one dimensions is specified, the other dimensions are assumed to be
        equal. This internally loads the texture into ram, splits it into smaller
        sub-images, and then calls the load_3d_texture from the Panda loader """

        tempfile_name = "/$$slice_loader_temp-" + str(time.time()) + "/"
        tile_size_y = tile_size_x if tile_size_y is None else tile_size_y
        num_tiles = tile_size_x if num_tiles is None else num_tiles

        # Load sliced image from disk
        tex_handle = cls.load_texture(fname)
        source = PNMImage()
        tex_handle.store(source)
        width = source.get_x_size()

        # Find slice properties
        num_cols = width // tile_size_x
        temp_img = PNMImage(tile_size_x, tile_size_y, source.get_num_channels(), source.get_maxval())

        # Construct a ramdisk to write the files to
        vfs = VirtualFileSystem.get_global_ptr()
        ramdisk = VirtualFileMountRamdisk()
        vfs.mount(ramdisk, tempfile_name, 0)

        # Extract all slices and write them to the virtual disk
        for z_slice in range(num_tiles):
            slice_x = (z_slice % num_cols) * tile_size_x
            slice_y = (z_slice // num_cols) * tile_size_y
            temp_img.copy_sub_image(source, 0, 0, slice_x, slice_y, tile_size_x, tile_size_y)
            temp_img.write(tempfile_name + str(z_slice) + ".png")

        # Load the de-sliced texture from the ramdisk
        texture_handle = cls.load_3d_texture(tempfile_name + "/#.png")
        vfs.unmount(ramdisk)

        return texture_handle
