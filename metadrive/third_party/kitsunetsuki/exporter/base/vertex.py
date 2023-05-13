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

import bpy
import mathutils  # make sure to "import bpy" before

from metadrive.third_party.kitsunetsuki.base.vertex import uv_equals, normal_equals


class VertexMixin(object):
    def get_sharp_vertices(self, mesh):
        results = []
        if mesh.use_auto_smooth:
            for edge in mesh.edges:
                if edge.use_edge_sharp:
                    for vertex_id in edge.vertices:
                        results.append(vertex_id)

        return results

    def get_tangent_bitangent(self, mesh):
        results = {}

        for uv_name, uv_layer in mesh.uv_layers.items():
            mesh.calc_tangents(uvmap=uv_name)
            results[uv_name] = []
            for i, loop in mesh.loops.items():
                results[uv_name].append(
                    (
                        mathutils.Vector(loop.tangent),
                        mathutils.Vector(loop.bitangent),
                        loop.bitangent_sign,
                    )
                )
            mesh.free_tangents()

        return results

    def can_share_vertex(self, mesh, vertex, loop_id, uv, normal):
        if not mesh.uv_layers:
            return True

        if not mesh.uv_layers.active:
            return True

        uv_loop = mesh.uv_layers.active.data[loop_id]
        # if uv_equals(uv_loop.uv.to_2d(), uv) and normal_equals(vertex.normal, normal):
        if uv_equals(uv_loop.uv.to_2d(), uv) and normal_equals(mesh.loops[loop_id].normal, normal):
            return True

        return False
