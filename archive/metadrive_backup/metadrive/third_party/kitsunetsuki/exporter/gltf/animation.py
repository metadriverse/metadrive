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
import copy
import decimal
import math

try:
    from collections.abc import Callable
except ImportError:
    from collections import Callable

from metadrive.third_party.kitsunetsuki.base.matrices import get_bone_matrix, quat_to_list

from . import spec


class AnimationMixin(object):
    def _make_sampler(self, path, input_id, bone):
        # transforms
        channel = self._buffer.add_channel(
            {
                'componentType': spec.TYPE_FLOAT,
                'type': 'VEC4' if path == 'rotation' else 'VEC3',
                'extras': {
                    'reference': path,
                },
            }
        )

        gltf_sampler = {
            'interpolation': 'LINEAR',
            'input': input_id,
            'output': channel['bufferView'],
            'extras': {
                'joint': bone.name,
            }
        }
        return gltf_sampler

    def make_action(self, node, armature, action):
        gltf_armature = None
        for i, gltf_node in enumerate(self._root['nodes']):
            if gltf_node['name'] == armature.name:
                gltf_armature = gltf_node
                break
        if not gltf_armature:
            gltf_armature = self.make_armature(node, armature)

        gltf_skin = None
        for i in gltf_armature['children']:
            gltf_node = self._root['nodes'][i]
            if 'skin' in gltf_node:
                gltf_skin = self._root['skins'][gltf_node['skin']]
                break

        if not gltf_skin:
            print('FAILED TO FIND GLTF SKIN')
            return

        # <-- animation
        gltf_animation = {
            'name': action.name if action else 'GLTF_ANIMATION',
            'channels': [],
            'samplers': [],
        }

        # setup bones
        gltf_channels = {}
        gltf_samplers = []
        for gltf_joint_id in gltf_skin['joints']:
            gltf_joint = self._root['nodes'][gltf_joint_id]
            bone = armature.data.bones[gltf_joint['name']]

            gltf_target = {}
            if gltf_joint_id is not None:
                gltf_target['node'] = gltf_joint_id

            # time or animation frame
            channel = self._buffer.add_channel(
                {
                    'componentType': spec.TYPE_FLOAT,
                    'type': 'SCALAR',
                    'extras': {
                        'reference': 'input',
                    },
                }
            )
            input_id = channel['bufferView']

            for path in ('rotation', 'scale', 'translation'):
                gltf_samplers.append(self._make_sampler(path, input_id, bone))

                gltf_channel = {
                    'sampler': len(gltf_samplers) - 1,
                    'target': copy.copy(gltf_target),
                    'extras': {
                        'joint': bone.name,
                    }
                }
                gltf_channel['target']['path'] = path
                gltf_channels['{}/{}'.format(bone.name, path)] = gltf_channel

        gltf_animation['channels'] = list(gltf_channels.values())
        gltf_animation['samplers'] = gltf_samplers

        # set animation data
        frame_start = bpy.context.scene.frame_start
        frame_end = bpy.context.scene.frame_end
        if action and armature.animation_data:
            armature.animation_data.action = action
            frame_start, frame_end = action.frame_range

        frame = float(frame_start)
        frame_int = None
        t = decimal.Decimal(0)
        fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
        dt = decimal.Decimal(1 / fps)
        t += dt
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

            # write bone matrices
            for gltf_joint_id in gltf_skin['joints']:
                gltf_joint = self._root['nodes'][gltf_joint_id]
                bone = armature.pose.bones[gltf_joint['name']]
                bone_matrix = self._transform(get_bone_matrix(bone, armature))

                rotation = quat_to_list(bone_matrix.to_quaternion())
                scale = list(bone_matrix.to_scale())
                translation = list(bone_matrix.to_translation())

                gltf_channel = gltf_channels['{}/{}'.format(bone.name, 'rotation')]
                gltf_sampler = gltf_samplers[gltf_channel['sampler']]
                self._buffer.write(gltf_sampler['output'], *rotation)

                gltf_channel = gltf_channels['{}/{}'.format(bone.name, 'scale')]
                gltf_sampler = gltf_samplers[gltf_channel['sampler']]
                self._buffer.write(gltf_sampler['output'], *scale)

                gltf_channel = gltf_channels['{}/{}'.format(bone.name, 'translation')]
                gltf_sampler = gltf_samplers[gltf_channel['sampler']]
                self._buffer.write(gltf_sampler['output'], *translation)

                self._buffer.write(gltf_sampler['input'], float(t))

            # advance to the next frame
            frame += speed_scale
            t += dt

        # animation -->

        self._root['animations'].append(gltf_animation)
