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
import bmesh


def obj2mesh(obj, triangulate=True):
    # convert the object to a mesh, so we can read the polygons
    # https://docs.blender.org/api/blender2.8/bpy.types.Object.html?highlight=to_mesh#bpy.types.Object.to_mesh
    # mesh = obj.to_mesh(
    #     bpy.context.depsgraph, apply_modifiers=True, calc_undeformed=True)
    mesh = obj.to_mesh()

    # get a BMesh representation
    b_mesh = bmesh.new()
    b_mesh.from_mesh(mesh)

    # triangulate the mesh
    if triangulate:
        bmesh.ops.triangulate(b_mesh, faces=b_mesh.faces)

    # copy the bmesh back to the original mesh
    b_mesh.to_mesh(mesh)

    # calculate the per-vertex normals, in case blender did not do that yet.
    mesh.calc_normals()
    mesh.calc_normals_split()

    return mesh
