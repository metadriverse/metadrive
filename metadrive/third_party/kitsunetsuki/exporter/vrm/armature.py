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

import math

from metadrive.third_party.kitsunetsuki.base.armature import is_left_bone, is_bone_matches
from metadrive.third_party.kitsunetsuki.base.objects import get_parent


class ArmatureMixin(object):
    def _make_vrm_bone(self, gltf_node_id, bone):
        vrm_bone = {
            'bone': None,
            'node': gltf_node_id,
            'useDefaultValues': True,
            'extras': {
                'name': bone.name,
            }
        }

        def is_hips(bone):
            return is_bone_matches(bone, ('hips', ))

        def is_upper_leg(bone, strict=True):
            names = ['thigh']
            if not strict:
                names.append('leg')
            is_upper = is_bone_matches(bone, names)
            is_child = is_hips(get_parent(bone))
            return is_upper or is_child

        def is_lower_leg(bone):
            is_lower = is_bone_matches(bone, ('calf', 'shin', 'knee'))
            is_child = is_upper_leg(get_parent(bone), strict=False)
            return is_lower or is_child

        def is_hand(bone):
            return is_bone_matches(bone, ('hand', 'wrist'))

        side = 'left' if is_left_bone(bone) else 'right'

        parents = []
        for i in range(1, 3 + 1):
            parent = get_parent(bone, i)
            if parent:
                parents.append(parent)

        if is_hips(bone):
            vrm_bone['bone'] = 'hips'

        elif (is_bone_matches(bone, ('upperchest', ))
              or (is_bone_matches(bone, ('spine', )) and is_hips(get_parent(bone, 3)))):
            vrm_bone['bone'] = 'upperChest'

        elif (is_bone_matches(bone, ('chest', )) or (is_bone_matches(bone,
                                                                     ('spine', )) and is_hips(get_parent(bone, 2)))):
            vrm_bone['bone'] = 'chest'

        elif is_bone_matches(bone, ('spine', )):
            vrm_bone['bone'] = 'spine'

        elif is_bone_matches(bone, ('neck', )):
            vrm_bone['bone'] = 'neck'

        elif is_bone_matches(bone, ('head', )):
            vrm_bone['bone'] = 'head'

        elif is_bone_matches(bone, ('eye', )):
            vrm_bone['bone'] = '{}Eye'.format(side)

        elif is_bone_matches(bone, ('foot', 'ankle')):
            vrm_bone['bone'] = '{}Foot'.format(side)

        elif is_lower_leg(bone):
            vrm_bone['bone'] = '{}LowerLeg'.format(side)

        elif is_upper_leg(bone):
            vrm_bone['bone'] = '{}UpperLeg'.format(side)

        elif is_bone_matches(bone, ('toe', )):
            vrm_bone['bone'] = '{}Toes'.format(side)

        elif is_bone_matches(bone, ('shoulder', 'clavicle')):
            vrm_bone['bone'] = '{}Shoulder'.format(side)

        elif is_bone_matches(bone, ('lowerarm', 'lower_arm', 'forearm', 'elbow')):
            vrm_bone['bone'] = '{}LowerArm'.format(side)

        elif is_bone_matches(bone, ('upperarm', 'upper_arm', 'arm')):
            vrm_bone['bone'] = '{}UpperArm'.format(side)

        elif any(map(is_hand, parents)):  # hand in parents -> finger
            if is_hand(get_parent(bone, 3)):  # 3 level deep parent
                part_name = 'Distal'
            elif is_hand(get_parent(bone, 2)):  # 2 level deep parent
                part_name = 'Intermediate'
            else:  # 1 level deep parent - direct parent
                part_name = 'Proximal'

            if is_bone_matches(bone, ('thumb', )):
                vrm_bone['bone'] = '{}Thumb{}'.format(side, part_name)

            elif is_bone_matches(bone, ('index', )):
                vrm_bone['bone'] = '{}Index{}'.format(side, part_name)

            elif is_bone_matches(bone, ('middle', )):
                vrm_bone['bone'] = '{}Middle{}'.format(side, part_name)

            elif is_bone_matches(bone, ('ring', )):
                vrm_bone['bone'] = '{}Ring{}'.format(side, part_name)

            elif is_bone_matches(bone, ('pinky', 'little')):
                vrm_bone['bone'] = '{}Little{}'.format(side, part_name)

        elif is_hand(bone):
            vrm_bone['bone'] = '{}Hand'.format(side)

        return vrm_bone

    def _make_vrm_spring(self, gltf_node_id, bone):
        vrm_spring = {
            'comment': bone.name,
            'stiffiness': 1,  # The resilience of the swaying object (the power of returning to the initial pose)
            'gravityPower': 0,
            'dragForce': 0,  # The resistance (deceleration) of automatic animation
            'gravityDir': {
                'x': 0,
                'y': -1,
                'z': 0,
            },  # down
            'center': -1,
            'hitRadius': 0,
            'bones': [gltf_node_id],
            # 'colliderGroups': [],
        }

        if bone.get('jiggle_stiffness', None) is not None:
            vrm_spring['stiffiness'] = bone.get('jiggle_stiffness')

        if bone.get('jiggle_gravity', None) is not None:
            vrm_spring['gravityPower'] = bone.get('jiggle_gravity')

        if bone.get('jiggle_amplitude', None) is not None:
            max_amp = 200
            jiggle_amplitude = min(max_amp, bone.get('jiggle_amplitude'))
            vrm_spring['dragForce'] = (max_amp - jiggle_amplitude) / max_amp

        return vrm_spring

    def make_armature(self, parent_node, armature):
        gltf_armature = super().make_armature(parent_node, armature)

        vrm_bones = set()
        vrm_springs = set()
        for bone_name, bone in armature.data.bones.items():
            gltf_node_id = None
            for gltf_node_id, gltf_node in enumerate(self._root['nodes']):
                if gltf_node['name'] == bone_name:
                    break
            else:
                continue

            vrm_bone = self._make_vrm_bone(gltf_node_id, bone)

            if vrm_bone['bone'] and vrm_bone['bone'] not in vrm_bones:
                vrm_bones.add(vrm_bone['bone'])
                self._root['extensions']['VRM']['humanoid']['humanBones'].append(vrm_bone)

                fp = self._root['extensions']['VRM']['firstPerson']

                if vrm_bone['bone'] == 'head':
                    fp['firstPersonBone'] = gltf_node_id
                    fp['extras'] = {'name': bone.name}

                elif vrm_bone['bone'] == 'leftEye':
                    fp.update(
                        {
                            'lookAtHorizontalOuter': {
                                'curve': [0, 0, 0, 1, 1, 1, 1, 0],
                                'xRange': 90,
                                'yRange': 10,
                            },
                            'lookAtHorizontalInner': {
                                'curve': [0, 0, 0, 1, 1, 1, 1, 0],
                                'xRange': 90,
                                'yRange': 10,
                            },
                            'lookAtVerticalDown': {
                                'curve': [0, 0, 0, 1, 1, 1, 1, 0],
                                'xRange': 90,
                                'yRange': 10,
                            },
                            'lookAtVerticalUp': {
                                'curve': [0, 0, 0, 1, 1, 1, 1, 0],
                                'xRange': 90,
                                'yRange': 10,
                            },
                        }
                    )

                    pose_bone = armature.pose.bones[bone_name]
                    for c in pose_bone.constraints:
                        if c.type == 'LIMIT_ROTATION':
                            fp['lookAtHorizontalOuter']['xRange'] = -math.degrees(c.min_x)
                            fp['lookAtHorizontalInner']['xRange'] = math.degrees(c.max_x)
                            fp['lookAtVerticalDown']['yRange'] = -math.degrees(c.min_z)
                            fp['lookAtVerticalUp']['yRange'] = math.degrees(c.max_z)
                            break

            pose_bone = armature.pose.bones[bone_name]

            # Wiggle Bones addon
            # https://blenderartists.org/t/wiggle-bones-a-jiggle-bone-implementation-for-2-8/1154726
            if pose_bone.get('jiggle_enable', False):
                # search for root bone
                while (pose_bone.parent and pose_bone.parent.get('jiggle_enable', False)):
                    pose_bone = pose_bone.parent

                if pose_bone.name not in vrm_springs:
                    vrm_spring = self._make_vrm_spring(gltf_node_id, pose_bone)
                    vrm_springs.add(pose_bone.name)
                    self._root['extensions']['VRM']['secondaryAnimation']['boneGroups'].append(vrm_spring)

        gltf_secondary = {
            'name': 'secondary',
            'translation': [0, 0, 0],
            'rotation': [0, 0, 0, 1],
            'scale': [1, 1, 1],
        }
        self._add_child(gltf_armature, gltf_secondary)

        return gltf_armature
