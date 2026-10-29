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

from . import spec


class VertexMixin(object):
    def make_vertex(
        self, obj_matrix, gltf_primitive, mesh, polygon, vertex, vertex_id, loop_id, use_smooth=False, can_merge=False
    ):
        # CO
        co = vertex.co
        if not self._z_up:
            co = self._matrix @ co
        if can_merge and not self._pose_freeze:
            co = obj_matrix @ co

        self._buffer.write(gltf_primitive['attributes']['POSITION'], *tuple(co))

        # normals
        # normal = vertex.normal if use_smooth else polygon.normal
        normal = mesh.loops[loop_id].normal if use_smooth else polygon.normal
        if not self._z_up:
            normal = self._matrix @ normal
        if can_merge and not self._pose_freeze:
            normal = obj_matrix.to_euler().to_matrix() @ normal

        self._buffer.write(gltf_primitive['attributes']['NORMAL'], *tuple(normal))

        # shape keys
        for i, sk_name in enumerate(gltf_primitive['extras']['targetNames']):
            sk_data = mesh.shape_keys.key_blocks[sk_name]
            sk_co = sk_data.data[vertex_id].co
            if not self._z_up:
                sk_co = self._matrix @ sk_co

            self._buffer.write(gltf_primitive['targets'][i]['POSITION'], *tuple(sk_co - co))

    def _write_uv(self, gltf_primitive, uv_id, u, v):
        texcoord = 'TEXCOORD_{}'.format(uv_id)
        if texcoord not in gltf_primitive['attributes']:
            channel = self._buffer.add_channel(
                {
                    'componentType': spec.TYPE_FLOAT,
                    'type': 'VEC2',
                    'extras': {
                        'reference': texcoord,
                    },
                }
            )
            gltf_primitive['attributes'][texcoord] = channel['bufferView']

        self._buffer.write(gltf_primitive['attributes'][texcoord], u, 1 - v)

    def _write_tbs(self, obj_matrix, gltf_primitive, t, b, s, can_merge=False):
        if not self._z_up:
            t = self._matrix @ t
        if can_merge and not self._pose_freeze:
            # t = obj_matrix @ t
            t = obj_matrix.to_euler().to_matrix() @ t
        x, y, z = t

        if 'TANGENT' not in gltf_primitive['attributes']:
            channel = self._buffer.add_channel(
                {
                    'componentType': spec.TYPE_FLOAT,
                    'type': 'VEC4',
                    'extras': {
                        'reference': 'TANGENT',
                    },
                }
            )
            gltf_primitive['attributes']['TANGENT'] = channel['bufferView']

        self._buffer.write(gltf_primitive['attributes']['TANGENT'], x, y, z, s)

    def _write_joints_weights(self, gltf_primitive, joints_num, joints_weights):
        for i, joint_weight in enumerate(joints_weights):
            # prepare joints buffer channel
            joints = 'JOINTS_{}'.format(i)
            if joints not in gltf_primitive['attributes']:
                if joints_num > 255:
                    ctype = spec.TYPE_UNSIGNED_SHORT
                else:
                    ctype = spec.TYPE_UNSIGNED_BYTE

                # Unity glTF importer (UniVRM/UniGLTF) compatibility
                if self._output.endswith('.vrm'):
                    ctype = spec.TYPE_UNSIGNED_SHORT

                channel = self._buffer.add_channel(
                    {
                        'componentType': ctype,
                        'type': 'VEC4',
                        'extras': {
                            'reference': joints,
                        },
                    }
                )
                gltf_primitive['attributes'][joints] = channel['bufferView']

            # write 4 joints
            keys = tuple(zip(*joint_weight))[0]
            assert len(keys) == 4
            self._buffer.write(gltf_primitive['attributes'][joints], *keys)

            # prepare weights buffer channel
            weights = 'WEIGHTS_{}'.format(i)
            if weights not in gltf_primitive['attributes']:
                channel = self._buffer.add_channel(
                    {
                        'componentType': spec.TYPE_FLOAT,
                        'type': 'VEC4',
                        'extras': {
                            'reference': weights,
                        },
                    }
                )
                gltf_primitive['attributes'][weights] = channel['bufferView']

            # write 4 weights
            values = tuple(zip(*joint_weight))[1]
            assert len(values) == 4
            self._buffer.write(gltf_primitive['attributes'][weights], *values)
