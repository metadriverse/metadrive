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

from panda3d.core import Vec3

from metadrive.render_pipeline.rpcore.pynative.rp_light import RPLight
from metadrive.render_pipeline.rpcore.pynative.shadow_source import ShadowSource


class RPPointLight(RPLight):
    """ Please refer to the native C++ implementation for docstrings and comments.
    This is just the python implementation, which does not contain documentation! """
    def __init__(self):
        RPLight.__init__(self, RPLight.LT_point_light)
        self._radius = 10.0
        self._inner_radius = 0.01

    def write_to_command(self, cmd):
        RPLight.write_to_command(self, cmd)
        cmd.push_float(self._radius)
        cmd.push_float(self._inner_radius)

    def set_radius(self, radius):
        self._radius = radius
        self.set_needs_update(True)
        self.invalidate_shadows()

    def get_radius(self):
        return self._radius

    radius = property(get_radius, set_radius)

    def set_inner_radius(self, inner_radius):
        assert inner_radius >= 0.01
        self._inner_radius = inner_radius
        self.set_needs_update(True)

    def get_inner_radius(self):
        return self._inner_radius

    inner_radius = property(get_inner_radius, set_inner_radius)

    def init_shadow_sources(self):
        for _ in range(6):
            self._shadow_sources.append(ShadowSource())

    def update_shadow_sources(self):
        directions = (Vec3(1, 0, 0), Vec3(-1, 0, 0), Vec3(0, 1, 0), Vec3(0, -1, 0), Vec3(0, 0, 1), Vec3(0, 0, -1))

        fov = 90.0 + 3.0
        for i, source in enumerate(self._shadow_sources):
            source.set_resolution(self.get_shadow_map_resolution())
            source.set_perspective_lens(fov, self._near_plane, self._radius, self._position, directions[i])
