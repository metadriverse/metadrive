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
import hashlib

from panda3d.core import PStatCollector, Mat4, Point4, Vec3
from metadrive.render_pipeline.rpcore.globals import Globals


def rgb_from_string(text, min_brightness=0.6):
    """ Creates a rgb color from a given string """
    ohash = hashlib.md5(text[::-1].encode("ascii")).hexdigest()
    r, g, b = int(ohash[0:2], 16), int(ohash[2:4], 16), int(ohash[4:6], 16)
    neg_inf = 1.0 - min_brightness
    return (
        min_brightness + r / 255.0 * neg_inf, min_brightness + g / 255.0 * neg_inf, min_brightness + b / 255.0 * neg_inf
    )


def profile(func):
    """ Handy decorator which can be used to profile a function with pstats """
    collector_name = "Debug:%s" % func.__name__

    global_showbase = Globals.base

    # Insert the collector to a custom dictionary attached to the base
    if hasattr(global_showbase, 'custom_collectors'):
        if collector_name in global_showbase.custom_collectors.keys():
            pstat = global_showbase.custom_collectors[collector_name]
        else:
            global_showbase.custom_collectors[collector_name] = \
                PStatCollector(collector_name)
            pstat = global_showbase.custom_collectors[collector_name]
    else:
        pstat = PStatCollector(collector_name)
        global_showbase.custom_collectors = {}
        global_showbase.custom_collectors[collector_name] = pstat

    def do_pstat(*args, **kargs):
        pstat.start()
        returned = func(*args, **kargs)
        pstat.stop()
        return returned

    do_pstat.__name__ = func.__name__
    do_pstat.__dict__ = func.__dict__
    do_pstat.__doc__ = func.__doc__
    return do_pstat


class profile_cpu(object):  # noqa # pylint: disable=invalid-name,too-few-public-methods
    """
    Context manager for profiling CPU duration. This is useful for timing
    loading of files or other CPU-heavy operations. Example usage:

      with profile_cpu("Some Task"):
        some_slow_operation()

    Duration of the process will be print out on the console later on.
    """
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.process_time()

    def __exit__(self, *args):
        duration = (time.process_time() - self.start_time) * 1000.0
        print(self.name, "took", round(duration, 2), "ms ")


def snap_shadow_map(mvp, cam_node, resolution):
    """ 'Snaps' a shadow map to make sure it always is on full texel centers.
    This ensures no flickering occurs while moving the shadow map.
    This works by projecting the Point (0,0,0) to light space, compute the
    texcoord differences and offset the light world space position by that. """
    mvp = Mat4(mvp)
    base_point = mvp.xform(Point4(0, 0, 0, 1)) * 0.5 + 0.5
    texel_size = 1.0 / float(resolution)
    offset_x = base_point.x % texel_size
    offset_y = base_point.y % texel_size
    mvp.invert_in_place()
    new_base = mvp.xform(
        Point4(
            (base_point.x - offset_x) * 2.0 - 1.0, (base_point.y - offset_y) * 2.0 - 1.0, (base_point.z) * 2.0 - 1.0, 1
        )
    )
    cam_node.set_pos(cam_node.get_pos() - Vec3(new_base.x, new_base.y, new_base.z))
