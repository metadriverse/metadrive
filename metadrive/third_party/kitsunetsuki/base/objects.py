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
import json


def get_object_properties(obj):
    text = bpy.data.texts.get(obj.name)
    if text:
        return json.loads(text.as_string() or '{}')
    else:
        return {}


def is_collision(obj):
    return obj.rigid_body is not None


def is_object_visible(obj):
    for collection in bpy.data.collections:
        if obj.name in collection.objects:
            if collection.hide_viewport:
                return False

    if obj.hide_viewport:
        return False

    return True


def set_active_object(obj):
    bpy.context.view_layer.objects.active = obj


def apply_modifiers(obj, triangulate=False, apply_scale=False):
    is_activated = False

    # if triangulate:
    #     bpy.ops.object.modifier_add(type='TRIANGULATE')
    #     obj.modifiers[-1].keep_custom_normals = True

    for mod in obj.modifiers:
        if not mod or not mod.show_viewport:
            continue

        if mod.type in ('ARMATURE', 'COLLISION'):
            continue

        if not is_activated:
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(state=True)
            set_active_object(obj)
            is_activated = True

        try:
            bpy.ops.object.modifier_apply(modifier=mod.name)
        except Exception as e:
            print(
                'FAILED TO APPLY MODIFIER {mod_name} [{mod_type}] ON OBJECT {obj_name}'.format(
                    **{
                        'mod_name': mod.name,
                        'mod_type': mod.type,
                        'obj_name': obj.name,
                    }
                )
            )
            raise e

    if apply_scale:
        if not is_activated:
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(state=True)
            set_active_object(obj)
            is_activated = True

        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def get_parent(obj, level=1):
    result = obj

    for i in range(level):
        if result is None:
            return None
        result = result.parent

    return result
