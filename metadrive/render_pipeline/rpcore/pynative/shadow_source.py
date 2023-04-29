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

from panda3d.core import Mat4, PerspectiveLens, LVector3, LVecBase2i, LVecBase2f
from panda3d.core import BoundingSphere


class ShadowSource(object):

    """ Please refer to the native C++ implementation for docstrings and comments.
    This is just the python implementation, which does not contain documentation! """

    def __init__(self):
        self._slot = -1
        self._needs_update = True
        self._resolution = 512
        self._mvp = 0.0
        self._region = LVecBase2i(-1)
        self._region_uv = LVecBase2f(0.0)
        self._bounds = BoundingSphere()

    def set_resolution(self, resolution):
        self._resolution = resolution

    def set_needs_update(self, flag):
        self._needs_update = flag

    def set_slot(self, slot):
        self._slot = slot

    def set_region(self, region, region_uv):
        self._region = region
        self._region_uv = region_uv

    def clear_region(self):
        self._region = LVecBase2i(-1)
        self._region_uv = LVecBase2f(0.0)

    def get_bounds(self):
        return self._bounds

    def set_perspective_lens(self, fov, near_plane, far_plane, pos, direction):
        transform_mat = Mat4.translate_mat(-pos)
        temp_lens = PerspectiveLens(fov, fov)
        temp_lens.set_film_offset(0, 0)
        temp_lens.set_near_far(near_plane, far_plane)
        temp_lens.set_view_vector(direction, LVector3.up())
        self.set_matrix_lens(transform_mat * temp_lens.get_projection_mat())

        hexahedron = temp_lens.make_bounds()
        center = (hexahedron.get_min() + hexahedron.get_max()) * 0.5
        self._bounds = BoundingSphere(pos + center, (hexahedron.get_max() - center).length())

    def set_matrix_lens(self, mvp):
        self._mvp = mvp
        self.set_needs_update(True)

    def has_region(self):
        return (self._region.x >= 0 and self._region.y >= 0 and
                self._region.z >= 0 and self._region.w >= 0)

    def has_slot(self):
        return self._slot >= 0

    def get_slot(self):
        return self._slot

    def get_needs_update(self):
        return not self.has_region or self._needs_update

    def get_resolution(self):
        return self._resolution

    def get_mvp(self):
        return self._mvp

    def get_region(self):
        return self._region

    def get_uv_region(self):
        return self._region_uv

    def write_to_command(self, cmd):
        cmd.push_mat4(self._mvp)
        cmd.push_vec4(self._region_uv)
