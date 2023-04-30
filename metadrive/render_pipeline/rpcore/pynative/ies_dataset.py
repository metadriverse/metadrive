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

from metadrive.render_pipeline.rplibs.six.moves import range  # pylint: disable=import-error

import math
from panda3d.core import PNMImage


class IESDataset(object):
    """ Please refer to the native C++ implementation for docstrings and comments.
    This is just the python implementation, which does not contain documentation! """
    def __init__(self):
        self._vertical_angles = None
        self._horizontal_angles = None
        self._candela_values = None

    def set_vertical_angles(self, vertical_angles):
        self._vertical_angles = vertical_angles

    def set_horizontal_angles(self, horizontal_angles):
        self._horizontal_angles = horizontal_angles

    def set_candela_values(self, candela_values):
        self._candela_values = candela_values

    def generate_dataset_texture_into(self, dest_tex, layer_index):
        resolution_vertical = dest_tex.get_y_size()
        resolution_horizontal = dest_tex.get_x_size()

        dest = PNMImage(resolution_vertical, resolution_horizontal, 1, 65535)

        for vert in range(resolution_vertical):
            for horiz in range(resolution_horizontal):
                vert_angle = vert / (resolution_vertical - 1.0)
                vert_angle = math.cos(vert_angle * math.pi) * 90.0 + 90.0
                horiz_angle = horiz / (resolution_horizontal - 1.0) * 360.0
                candela = self.get_candela_value(vert_angle, horiz_angle)
                dest.set_xel(vert, horiz, candela)

        dest_tex.load(dest, layer_index, 0)

    def get_candela_value(self, vertical_angle, horizontal_angle):  # noqa # pylint: disable=unused-argument
        # NOTICE: Since python is slower, we always only assume a dataset without
        # horizontal angles. This still produces convincing results, but does
        # generate much faster.
        return self.get_vertical_candela_value(0, vertical_angle)

    def get_candela_value_from_index(self, vertical_angle_idx, horizontal_angle_idx):
        index = vertical_angle_idx + horizontal_angle_idx * len(self._vertical_angles)
        return self._candela_values[index]

    def get_vertical_candela_value(self, horizontal_angle_idx, vertical_angle):
        if vertical_angle < 0.0:
            return 0.0

        if vertical_angle > self._vertical_angles[len(self._vertical_angles) - 1]:
            return 0.0

        for vertical_index in range(1, len(self._vertical_angles)):
            curr_angle = self._vertical_angles[vertical_index]

            if curr_angle > vertical_angle:
                prev_angle = self._vertical_angles[vertical_index - 1]
                prev_value = self.get_candela_value_from_index(vertical_index - 1, horizontal_angle_idx)
                curr_value = self.get_candela_value_from_index(vertical_index, horizontal_angle_idx)
                lerp = (vertical_angle - prev_angle) / (curr_angle - prev_angle)
                assert lerp >= 0.0 and lerp <= 1.0
                return curr_value * lerp + prev_value * (1.0 - lerp)
        return 0.0
