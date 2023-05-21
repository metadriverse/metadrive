# Copyright (c) 2020 kitsune.ONE team.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


def uv_equals(uv1, uv2):
    u1, v1 = uv1
    u2, v2 = uv2

    if abs(u1 - u2) > 0.001:
        return False

    if abs(v1 - v2) > 0.001:
        return False

    return True


def normal_equals(n1, n2):
    x1, y1, z1 = n1
    x2, y2, z2 = n2

    if abs(x1 - x2) > 0.001:
        return False

    if abs(y1 - y2) > 0.001:
        return False

    if abs(z1 - z2) > 0.001:
        return False

    return True
