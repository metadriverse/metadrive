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
from metadrive.render_pipeline.rplibs.six.moves import range  # pylint: disable=import-error

import struct


class GPUCommand(object):
    """ Please refer to the native C++ implementation for docstrings and comments.
    This is just the python implementation, which does not contain documentation! """

    CMD_invalid = 0
    CMD_store_light = 1
    CMD_remove_light = 2
    CMD_store_source = 3
    CMD_remove_sources = 4

    def __init__(self, command_type):
        self._command_type = command_type
        self._current_index = 0
        self._data = [0.0] * 32
        self.push_int(command_type)

    def push_int(self, value):
        self.push_float(float(value))

    def push_float(self, value):
        if self._current_index >= 32:
            print("GPUCommand: out of bounds!")
            return
        self._data[self._current_index] = float(value)
        self._current_index += 1

    def push_vec3(self, value):
        self.push_float(value.x)
        self.push_float(value.y)
        self.push_float(value.z)

    def push_vec4(self, value):
        self.push_vec3(value)
        self.push_float(value.get_w())

    def push_mat4(self, value):
        for i in range(4):
            for j in range(4):
                self.push_float(value.get_cell(i, j))

    @staticmethod
    def get_uses_integer_packing():
        return False

    def write_to(self, dest, command_index):
        data = struct.pack("f" * 32, *self._data)
        offset = command_index * 32 * 4
        dest.set_subdata(offset, 32 * 4, data)

    def write(self, out=None):  # pylint: disable=unused-argument
        print("GPUCommand(type=", self._command_type, "size=", self._current_index, ")")
        print("Data:", ', '.join([str(i) for i in self._data]))
