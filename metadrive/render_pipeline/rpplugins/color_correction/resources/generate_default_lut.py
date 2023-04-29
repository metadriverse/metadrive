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
from __future__ import division, print_function

from panda3d.core import PNMImage

lut_size = 64
lut_cols = 8
lut_rows = (lut_size + lut_cols - 1) // lut_cols

img = PNMImage(lut_size * lut_cols, lut_size * lut_rows, 3, 2**16 - 1)


def to_linear(v):
    return float(v) / float(lut_size - 1)


def to_linear_inv(v):
    return 1 - to_linear(v)

for r in range(lut_size):
    for g in range(lut_size):
        for b in range(lut_size):
            slice_offset_x = (b % lut_cols) * lut_size
            slice_offset_y = (b // lut_cols) * lut_size
            img.set_xel(r + slice_offset_x, g + slice_offset_y,
                        to_linear(r), to_linear_inv(g), to_linear(b))

img.write("default_lut.png")
