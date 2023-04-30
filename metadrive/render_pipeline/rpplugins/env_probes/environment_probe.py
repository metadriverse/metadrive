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

import struct
from panda3d.core import TransformState, Vec3, Mat4, BoundingSphere

from metadrive.render_pipeline.rpcore.rpobject import RPObject


class EnvironmentProbe(RPObject):
    """ Simple class, representing an environment probe """
    def __init__(self):
        """ Inits a new environment probe """
        RPObject.__init__(self)
        self.index = -1
        self.last_update = -1
        self._transform = TransformState.make_identity()
        self._bounds = BoundingSphere(Vec3(0), 1.0)
        self._modified = True
        self._parallax_correction = True
        self._border_smoothness = 0.1

    @property
    def modified(self):
        """ Returns whether the probe was modified since the last write """
        return self._modified

    @property
    def bounds(self):
        """ Returns the probes worldspace bounds """
        return self._bounds

    @property
    def parallax_correction(self):
        """ Returns whether parallax correction is enabled for this probe """
        return self._parallax_correction

    @parallax_correction.setter
    def parallax_correction(self, value):
        """ Sets whether parallax correction is enabled for this probe """
        self._parallax_correction = value
        self._modified = True

    @property
    def border_smoothness(self):
        """ Returns the border smoothness factor """
        return self._border_smoothness

    @border_smoothness.setter
    def border_smoothness(self, value):
        """ Sets the border smoothness factor """
        self._border_smoothness = value
        self._modified = True

    def set_pos(self, *args):
        """ Sets the probe position """
        self._transform = self._transform.set_pos(Vec3(*args))
        self.update_bounds()

    def set_hpr(self, *args):
        """ Sets the probe rotation """
        self._transform = self._transform.set_hpr(Vec3(*args))
        self.update_bounds()

    def set_scale(self, *args):
        """ Sets the probe scale """
        self._transform = self._transform.set_scale(Vec3(*args))
        self.update_bounds()

    def set_mat(self, matrix):
        """ Sets the probes matrix, overrides all other transforms """
        self._transform = TransformState.make_mat(matrix)
        self.update_bounds()

    def update_bounds(self):
        """ Updates the spheres bounds """
        mat = self._transform.get_mat()
        mid_point = mat.xform_point(Vec3(0, 0, 0))
        max_point = mat.xform_point(Vec3(1, 1, 1))
        radius = (mid_point - max_point).length()
        self._bounds = BoundingSphere(mid_point, radius)
        self._modified = True

    @property
    def matrix(self):
        """ Returns the matrix of the probe """
        return self._transform.get_mat()

    def write_to_buffer(self, buffer_ptr):
        """ Writes the probe to a given byte buffer """
        data, mat = [], Mat4(self._transform.get_mat())
        mat.invert_in_place()
        for i in range(4):
            for j in range(3):
                data.append(mat.get_cell(i, j))
        data += (float(self.index), 1.0 if self._parallax_correction else 0.0, self._border_smoothness, 0)
        data.append(self._bounds.get_center().x)
        data.append(self._bounds.get_center().y)
        data.append(self._bounds.get_center().z)
        data.append(self._bounds.get_radius())
        byte_data = struct.pack("f" * 20, *data)

        # 4 = sizeof float, 20 = floats per cubemap
        bytes_per_probe = 4 * 20
        buffer_ptr.set_subdata(self.index * bytes_per_probe, bytes_per_probe, byte_data)

        self._modified = False
