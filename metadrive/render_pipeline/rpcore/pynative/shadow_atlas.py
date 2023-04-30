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

from __future__ import print_function, division
from metadrive.render_pipeline.rplibs.six.moves import range  # pylint: disable=import-error

from panda3d.core import LVecBase4i, LVecBase4


class ShadowAtlas(object):
    """ Please refer to the native C++ implementation for docstrings and comments.
    This is just the python implementation, which does not contain documentation! """
    def __init__(self, size, tile_size=32):
        self._size = size
        self._tile_size = tile_size
        self._num_used_tiles = 0
        self.init_tiles()

    def init_tiles(self):
        self._num_tiles = self._size // self._tile_size

        def row():
            return [False for i in range(self._num_tiles)]  # pylint: disable=unused-variable

        self._flags = [row() for j in range(self._num_tiles)]  # pylint: disable=unused-variable

    def get_num_used_tiles(self):
        return self._num_used_tiles

    num_used_tiles = property(get_num_used_tiles)

    def get_coverage(self):
        return self._num_used_tiles / float(self._num_tiles**2)

    coverage = property(get_coverage)

    def reserve_region(self, x, y, w, h):
        self._num_used_tiles += w * h
        for x_offset in range(w):
            for y_offset in range(h):
                self._flags[x + x_offset][y + y_offset] = True

    def find_and_reserve_region(self, tile_width, tile_height):
        for x in range(self._num_tiles - tile_height + 1):
            for y in range(self._num_tiles - tile_width + 1):
                if self.region_is_free(x, y, tile_width, tile_height):
                    self.reserve_region(x, y, tile_width, tile_height)
                    return LVecBase4i(x, y, tile_width, tile_height)
        print("Failed to find a free region of size", tile_width, "x", tile_height)
        return LVecBase4i(-1)

    def free_region(self, region):
        self._num_used_tiles -= region.z * region.w
        for x in range(region.z):
            for y in range(region.w):
                self._flags[region.x + x][region.y + y] = False

    def get_tile_size(self):
        return self._tile_size

    def region_is_free(self, x, y, w, h):
        for x_offset in range(w):
            for y_offset in range(h):
                if self._flags[x + x_offset][y + y_offset]:
                    return False
        return True

    def get_required_tiles(self, resolution):
        if resolution % self._tile_size != 0:
            print("ShadowAtlas: Invalid atlas resolution!")
            return
        return resolution // self._tile_size

    def region_to_uv(self, region):
        flt = LVecBase4(region.x, region.y, region.z, region.w)
        return flt * (self._tile_size / self._size)
