#!/usr/bin/env python3
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

import argparse
import json
import struct


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, help='Input .gltf file path.')
    parser.add_argument('--extras', action='store_true', required=False, help="Shows node's extras.")

    return parser.parse_args()


def print_node(gltf_data, node_id, joints=None, skeletons=None, indent=1, parent_node=None, extras=None):
    gltf_node = gltf_data['nodes'][node_id]

    type_ = 'N'
    extra = ''

    matrix = ''
    if 'translation' in gltf_node:
        if sum(gltf_node['translation']) != 0:
            matrix += 'T'
    if 'rotation' in gltf_node:
        if not (gltf_node['rotation'][0] == 0 and gltf_node['rotation'][1] == 0 and gltf_node['rotation'][2] == 0
                and gltf_node['rotation'][3] == 1):
            matrix += 'R'
    if 'scale' in gltf_node:
        if not (gltf_node['scale'][0] == 1 and gltf_node['scale'][1] == 1 and gltf_node['scale'][2] == 1):
            matrix += 'S'
    if 'matrix' in gltf_node:
        matrix += 'M'

    if matrix:
        extra += ' <{}>'.format(matrix)

    if node_id in (skeletons or []):
        type_ = 'S'  # skeleton/armature
    elif node_id in (joints or []):
        type_ = 'J'  # joint/bone

    if 'mesh' in gltf_node:
        type_ = 'M'  # mesh/geometry

    if 'skin' in gltf_node:
        refs = []

        if 'skin' in gltf_node:
            skin_id = gltf_node['skin']
            gltf_skin = gltf_data['skins'][skin_id]
            v = '{} ({} joints)'.format(gltf_skin.get('name', 'SKIN #{}'.format(skin_id)), len(gltf_skin['joints']))
            refs.append(('skin', v))

            if 'skeleton' in gltf_skin:
                skeleton_id = gltf_skin['skeleton']
                gltf_skeleton = gltf_data['nodes'][skeleton_id]
                v = '{}'.format(gltf_skeleton.get('name', 'SKELETON #{}'.format(skeleton_id)))
                refs.append(('skeleton', v))

        if 'mesh' in gltf_node:
            mesh_id = gltf_node['mesh']
            gltf_mesh = gltf_data['meshes'][mesh_id]
            refs.append(('mesh', gltf_mesh['name']))

        extra += ' {' + ', '.join(['{}: {}'.format(*i) for i in refs]) + '}'

    if 'VRM' in (gltf_data.get('extensions') or {}):
        vrm_extra = []

        vrm_bones = gltf_data['extensions']['VRM']['humanoid']['humanBones']
        for vrm_bone in vrm_bones:
            if node_id == vrm_bone['node']:
                vrm_extra.append('VRM bone: {}'.format(vrm_bone['bone']))

        for group in gltf_data['extensions']['VRM']['secondaryAnimation']['boneGroups']:
            if node_id in group['bones']:
                vrm_extra.append('bonegroup')
                break

        if vrm_extra:
            extra += ' {%s}' % ', '.join(vrm_extra)

    # for child_node_id in gltf_node.get('children', []):
    #     child_gltf_node = gltf_data['nodes'][child_node_id]
    #     if 'skin' in child_gltf_node:
    #         # type_ = 'S'  # skeleton/armature
    #         skin_id = child_gltf_node['skin']
    #         gltf_skin = gltf_data['skins'][skin_id]
    #         # joints = gltf_skin['joints']

    is_ = ''
    for i in range(indent):
        if i < indent - 1:
            is_ += '  |'
        else:
            is_ += '  +'
    print('{} [{}] {}{}'.format(is_, type_, gltf_node['name'], extra))

    if extras:
        for k, v in gltf_node.get('extras', {}).items():
            print('   {}  {}: {}'.format(is_, k, v))

    for child_node_id in gltf_node.get('children', []):
        print_node(
            gltf_data, child_node_id, joints=joints, skeletons=skeletons, indent=indent + 1, parent_node=gltf_node
        )


def print_scene(gltf_data, scene_id, extras=False):
    gltf_scene = gltf_data['scenes'][scene_id]
    print(' [R] {}'.format(gltf_scene.get('name', 'SCENE')))

    # child to parent mapping
    parents = {}
    for parent_id, gltf_node in enumerate(gltf_data['nodes']):
        for child_id in gltf_node.get('children', ()):
            parents[child_id] = parent_id

    skeletons = set()
    joints = set()
    for gltf_skin in gltf_data['skins']:
        joints |= set(gltf_skin['joints'])
        # search for the root bone
        for joint_id in gltf_skin['joints']:
            if parents.get(joint_id) not in gltf_skin['joints']:  # no parent bone
                skeletons.add(joint_id)  # mark root bone as skeleton
                break

    for node_id in gltf_scene['nodes']:
        print_node(gltf_data, node_id, joints=joints, skeletons=skeletons, extras=extras)


def print_anim(gltf_data, gltf_anim):
    extra = ''
    input_id = gltf_anim['samplers'][0]['input']
    extra += '{} frames'.format(gltf_data['accessors'][input_id]['count'])
    if extra:
        extra = ' {' + extra + '}'

    print(' [A] {}{}'.format(gltf_anim['name'], extra))


def print_mat(gltf_data, gltf_mat):
    tex_ids = []
    if 'baseColorTexture' in gltf_mat.get('pbrMetallicRoughness', {}):
        tex_ids.append(('Color', gltf_mat['pbrMetallicRoughness']['baseColorTexture']['index']))
    if 'metallicRoughnessTexture' in gltf_mat.get('pbrMetallicRoughness', {}):
        tex_ids.append(('MetRough', gltf_mat['pbrMetallicRoughness']['metallicRoughnessTexture']['index']))
    if 'normalTexture' in gltf_mat:
        tex_ids.append(('Norm', gltf_mat['normalTexture']['index']))
    if 'emissiveTexture' in gltf_mat:
        tex_ids.append(('Emit', gltf_mat['emissiveTexture']['index']))

    print(' [M] {}'.format(gltf_mat['name']))
    for tex_type, tex_id in tex_ids:
        print_tex(gltf_data, tex_type, gltf_data['textures'][tex_id])


def print_tex(gltf_data, gltf_tex_type, gltf_tex):
    sampler = gltf_data['samplers'][gltf_tex['sampler']]
    # source = gltf_data['images'][gltf_tex['source']]
    print('  + [T] {name} <{type}>'.format(**{
        'type': gltf_tex_type,
        'name': sampler.get('name', 'SAMPLER'),
    }))


def main():
    args = parse_args()
    gltf_data = {
        'scene': {},
    }

    if args.input.endswith('.glb') or args.input.endswith('.vrm'):
        with open(args.input, 'rb') as f:
            assert f.read(4) == b'glTF'  # header
            assert struct.unpack('<I', f.read(4))[0] == 2  # version
            full_size = struct.unpack('<I', f.read(4))

            chunk_type = None
            chunk_data = None
            while True:
                chunk_size = struct.unpack('<I', f.read(4))[0]
                chunk_type = f.read(4)
                chunk_data = f.read(chunk_size)
                if chunk_type == b'JSON':
                    break

            if chunk_type == b'JSON':
                gltf_data = json.loads(chunk_data)

    else:
        with open(args.input, 'r') as f:
            gltf_data = json.load(f)

    if 'generator' in gltf_data.get('asset', {}):
        print('glTF generator: {}'.format(gltf_data['asset']['generator']))

    if 'VRM' in gltf_data.get('extensions', {}):
        vrm_meta = gltf_data['extensions']['VRM']
        print('VRM exporter: {}'.format(vrm_meta['exporterVersion']))

    print_scene(gltf_data, gltf_data['scene'], extras=args.extras)

    for gltf_anim in (gltf_data.get('animations') or []):
        print_anim(gltf_data, gltf_anim)

    for gltf_mat in (gltf_data.get('materials') or []):
        print_mat(gltf_data, gltf_mat)


if __name__ == '__main__':
    main()
