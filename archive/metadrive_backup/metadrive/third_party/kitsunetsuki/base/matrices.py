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
import itertools


def quat_to_list(quat):
    return [quat.x, quat.y, quat.z, quat.w]


def matrix_to_list(matrix):
    return list(itertools.chain(*map(tuple, matrix.col)))


def get_bone_matrix_local(bone):
    if isinstance(bone, bpy.types.PoseBone):
        return bone.matrix
    if isinstance(bone, bpy.types.EditBone):
        return bone.matrix
    elif isinstance(bone, bpy.types.Bone):
        return bone.matrix_local


def get_bone_matrix(bone, armature):
    bone_matrix = get_bone_matrix_local(bone)
    if bone.parent:
        parent_bone_matrix = get_bone_matrix_local(bone.parent)
        return parent_bone_matrix.inverted() @ bone_matrix
    else:  # root bone
        return bone_matrix


def get_object_matrix(obj, armature=None):
    if armature is None:
        return obj.matrix_local
    else:  # skinned mesh/object
        return (obj.matrix_world @ armature.matrix_world @ armature.matrix_world.inverted())


def get_inverse_bind_matrix(bone, armature):
    return bone.matrix_local.inverted() @ armature.matrix_world.inverted()
