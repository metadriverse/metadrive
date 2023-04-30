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

from panda3d.core import Vec3, Mat3


def color_from_temperature(temperature):
    # Thanks to rdb for this conversion script
    mm = 1000.0 / temperature  # pylint: disable=invalid-name
    mm2 = mm**2
    mm3 = mm2 * mm
    x, y = 0, 0

    if temperature < 4000:
        x = -0.2661239 * mm3 - 0.2343580 * mm2 + 0.8776956 * mm + 0.179910
    else:
        x = -3.0258469 * mm3 + 2.1070379 * mm2 + 0.2226347 * mm + 0.240390

    x2 = x**2  # pylint: disable=invalid-name
    x3 = x2 * x  # pylint: disable=invalid-name
    if temperature < 2222:
        y = -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683
    elif temperature < 4000:
        y = -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483

    # xyY to XYZ, assuming Y=1.
    xyz = Vec3(x / y, 1, (1 - x - y) / y)

    # Convert XYZ to linearized sRGB.
    xyz_to_rgb = Mat3(3.2406, -0.9689, 0.0557, -1.5372, 1.8758, -0.2050, -0.4986, 0.0415, 1.0570)

    return xyz_to_rgb.xform(xyz)


__all__ = ["RPLight"]


class RPLight(object):
    """ Please refer to the native C++ implementation for docstrings and comments.
    This is just the python implementation, which does not contain documentation! """

    LT_empty = 0
    LT_point_light = 1
    LT_spot_light = 2

    def __init__(self, light_type):
        self._light_type = light_type
        self._needs_update = False
        self._casts_shadows = False
        self._slot = -1
        self._position = Vec3(0)
        self._color = Vec3(1)
        self._ies_profile = -1
        self._source_resolution = 512
        self._near_plane = 0.5
        self._energy = 20
        self._shadow_sources = []

    def get_num_shadow_sources(self):
        return len(self._shadow_sources)

    def get_shadow_source(self, index):
        return self._shadow_sources[index]

    def clear_shadow_sources(self):
        self._shadow_sources = []

    def set_needs_update(self, flag):
        self._needs_update = flag

    def get_needs_update(self):
        return self._needs_update

    def has_slot(self):
        return self._slot >= 0

    def get_slot(self):
        return self._slot

    def remove_slot(self):
        self._slot = -1

    def assign_slot(self, slot):
        self._slot = slot

    def invalidate_shadows(self):
        for source in self._shadow_sources:
            source.set_needs_update(True)

    def set_pos(self, *args):
        self._position = Vec3(*args)
        self.set_needs_update(True)
        self.invalidate_shadows()

    def get_pos(self):
        return self._position

    pos = property(get_pos, set_pos)

    def set_color(self, *args):
        self._color = Vec3(*args)
        self._color /= (0.2126 * self._color.x + 0.7152 * self._color.y + 0.0722 * self._color.z)
        self.set_needs_update(True)

    def set_color_from_temperature(self, temperature):
        self.set_color(color_from_temperature(temperature))

    def get_color(self):
        return self._color

    color = property(get_color, set_color)

    def set_energy(self, energy):
        self._energy = energy

    def get_energy(self):
        return self._energy

    energy = property(get_energy, set_energy)

    def get_light_type(self):
        return self._light_type

    light_type = property(get_light_type)

    def set_casts_shadows(self, flag):
        if self.has_slot():
            print("Light is already attached!")
            return
        self._casts_shadows = flag

    def get_casts_shadows(self):
        return self._casts_shadows

    casts_shadows = property(get_casts_shadows, set_casts_shadows)

    def set_shadow_map_resolution(self, resolution):
        self._source_resolution = resolution
        self.invalidate_shadows()

    def get_shadow_map_resolution(self):
        return self._source_resolution

    shadow_map_resolution = property(get_shadow_map_resolution, set_shadow_map_resolution)

    def set_ies_profile(self, profile):
        self._ies_profile = profile
        self.set_needs_update(True)

    def get_ies_profile(self):
        return self._ies_profile

    def has_ies_profile(self):
        return self._ies_profile >= 0

    def clear_ies_profile(self):
        self.set_ies_profile(-1)

    ies_profile = property(get_ies_profile, set_ies_profile)

    def set_near_plane(self, near_plane):
        self._near_plane = near_plane
        self.invalidate_shadows()

    def get_near_plane(self):
        return self._near_plane

    near_plane = property(get_near_plane, set_near_plane)

    def write_to_command(self, cmd):
        cmd.push_int(self._light_type)
        cmd.push_int(self._ies_profile)

        if self._casts_shadows:
            cmd.push_int(self._shadow_sources[0].get_slot())
        else:
            cmd.push_int(-1)

        cmd.push_vec3(self._position)
        cmd.push_vec3(self._color * self._energy / 100.0)
