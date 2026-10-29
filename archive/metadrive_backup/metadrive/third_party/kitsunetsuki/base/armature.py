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


def get_armature(obj):
    parent = obj.parent
    while parent:
        if parent.type == 'ARMATURE':
            return parent
        parent = parent.parent


def is_left_bone(bone):
    return (
        bone.name.endswith('_L') or bone.name.endswith('.L') or bone.name.lower().startswith('left')
        or '.L.' in bone.name or '_L.' in bone.name or ':Left' in bone.name
    )


def is_bone_matches(bone, names):
    if bone is None:
        return False

    if bone.name.lower().endswith('_end'):
        return False

    for name in names:
        if name in bone.name.lower():
            return True

    return False
