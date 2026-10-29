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

from metadrive.third_party.kitsunetsuki.base.matrices import get_object_matrix

from panda3d.egg import EggVertex, EggVertexUV


class VertexMixin(object):
    def _get_uv_name(self, uv_layer):
        if uv_layer.active:
            return ''
        else:
            return uv_layer.name.replace(' ', '_')

    def make_vertex(self, parent_obj_matrix, obj_matrix, polygon, vertex, use_smooth=False, can_merge=False):
        egg_vertex = EggVertex()
        egg_vertex.set_color((1, 1, 1, 1))

        if can_merge:
            co = obj_matrix @ vertex.co
        else:
            co = vertex.co

        egg_vertex.set_pos(tuple(co))

        if use_smooth:
            normal = parent_obj_matrix @ vertex.normal
        else:
            normal = parent_obj_matrix @ polygon.normal

        egg_vertex.set_normal(tuple(normal))

        return egg_vertex

    def make_vertex_uv(self, uv_layer, uv):
        egg_vertex_uv = EggVertexUV(self._get_uv_name(uv_layer), tuple(uv))
        return egg_vertex_uv
