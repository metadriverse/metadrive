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
import collections
import math

try:
    from collections.abc import Callable
except ImportError:
    from collections import Callable

from metadrive.third_party.kitsunetsuki.base.matrices import get_bone_matrix

from panda3d.core import CS_zup_right
from panda3d.egg import EggTable, EggXfmSAnim


class AnimationMixin(object):
    def make_action(self, node, armature, action):
        # <-- root table
        egg_root_table = EggTable('')

        # <-- animation bundle
        egg_animation = EggTable('Armature')
        egg_animation.set_table_type(EggTable.TT_bundle)

        # <-- skeleton table
        egg_skeleton = EggTable('<skeleton>')

        # setup bones
        egg_joints = {}
        egg_joints_anims = {}
        for bone_name, bone in armature.data.bones.items():
            # <-- joint table
            egg_joint = EggTable(bone_name)

            # <-- xfm_anim_s
            egg_xfm_anim_s = EggXfmSAnim('xform', CS_zup_right)
            # set order from glTF
            # i, j, k - scale -> s
            # h, p, r - rotation
            # x, y, z - location -> ?
            # egg_xfm_anim_s.set_order('shprxyz')
            egg_xfm_anim_s.set_fps(bpy.context.scene.render.fps / bpy.context.scene.render.fps_base)
            egg_joint.add_child(egg_xfm_anim_s)
            egg_joints_anims[bone_name] = egg_xfm_anim_s
            # xfm_anim_s -->

            if bone.parent:
                egg_joints[bone.parent.name].add_child(egg_joint)
            else:  # root bone
                egg_skeleton.add_child(egg_joint)

            egg_joints[bone_name] = egg_joint
            # joint table -->

        # set animation data
        frame_start = bpy.context.scene.frame_start
        frame_end = bpy.context.scene.frame_end
        if action:
            armature.animation_data.action = action
            frame_start, frame_end = action.frame_range

        frame = float(frame_start)
        frame_int = None
        while frame <= frame_end:
            # switch frame
            if frame_int != math.floor(frame):
                frame_int = math.floor(frame)
                bpy.context.scene.frame_current = frame_int
                bpy.context.scene.frame_set(frame_int)

            if isinstance(self._speed_scale, Callable):
                speed_scale = self._speed_scale(frame_int)
            else:
                speed_scale = self._speed_scale

            # switch subframe
            if speed_scale != 1:
                bpy.context.scene.frame_subframe = frame - frame_int

            # save bone matrices
            for bone_name, bone in armature.pose.bones.items():
                s_anim = egg_joints_anims[bone_name]
                bone_matrix = get_bone_matrix(bone, armature)

                # egg_joints_anims[bone_name].add_data(matrix_to_panda(bone_matrix))

                i, j, k = bone_matrix.to_scale()
                s_anim.add_component_data('i', i)
                s_anim.add_component_data('j', j)
                s_anim.add_component_data('k', k)

                # if bone.parent:
                #     p, r, h = tuple(map(math.degrees, bone_matrix.to_euler()))  # YABEE
                #     s_anim.add_component_data('h', h)
                #     s_anim.add_component_data('p', p)
                #     s_anim.add_component_data('r', r)
                # else:
                #     h, r, p = tuple(map(math.degrees, bone_matrix.to_euler('XZY')))
                #     s_anim.add_component_data('h', -h)
                #     s_anim.add_component_data('p', p)
                #     s_anim.add_component_data('r', r)

                # hpr == zxy ?
                p, r, h = tuple(map(math.degrees, bone_matrix.to_euler('YXZ')))
                s_anim.add_component_data('h', h)
                s_anim.add_component_data('p', p)
                s_anim.add_component_data('r', r)

                x, y, z = bone_matrix.to_translation()
                s_anim.add_component_data('x', x)
                s_anim.add_component_data('y', y)
                s_anim.add_component_data('z', z)

            # advance to the next frame
            frame += speed_scale

        egg_animation.add_child(egg_skeleton)
        # skeleton table -->

        egg_root_table.add_child(egg_animation)
        # animation bundle -->

        # root table -->

        node.add_child(egg_root_table)
