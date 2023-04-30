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

import os
import sys
from panda3d.core import *
from direct.showbase.ShowBase import ShowBase


def _error(msg):
    print("\n" * 4, file=sys.stderr)
    print("ERROR:", msg, file=sys.stderr)
    print("\n" * 4, file=sys.stderr)
    sys.exit(1)


class Application(ShowBase):
    def __init__(self):
        load_prc_file_data(
            "", """
            window-type offscreen
            win-size 100 100
            color-bits 0
            depth-bits 0
            back-buffers 0
            print-pipe-types #f
        """
        )
        ShowBase.__init__(self)

        if not self.win.gsg.supports_compute_shaders:
            _error("Compute shaders not supported! Please update your driver, or get a newer gpu.")

        print("All checks passed successfully")


Application()
