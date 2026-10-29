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

import io
import os
import struct

from . import spec


class GLTFBuffer(object):
    def __init__(self, filepath):
        self._filepath = filepath
        self._channels = []
        self._metadata = []

    def add_channel(self, metadata):
        self._channels.append(io.BytesIO())
        self._metadata.append(metadata)
        self._metadata[-1]['bufferView'] = len(self._metadata) - 1
        self._metadata[-1]['count'] = 0
        return self._metadata[-1]

    def write(self, channel_id, *values):
        size = {
            'SCALAR': 1,
            'VEC2': 2,
            'VEC3': 3,
            'VEC4': 4,
            'MAT4': 4 * 4,
        }[self._metadata[channel_id]['type']]
        assert size == len(values)

        if 'min' not in self._metadata[channel_id]:
            self._metadata[channel_id]['min'] = [None] * size

        if 'max' not in self._metadata[channel_id]:
            self._metadata[channel_id]['max'] = [None] * size

        for i, value in enumerate(values):
            if self._metadata[channel_id]['min'][i] is None:
                self._metadata[channel_id]['min'][i] = value
            else:
                self._metadata[channel_id]['min'][i] = min(self._metadata[channel_id]['min'][i], value)

            if self._metadata[channel_id]['max'][i] is None:
                self._metadata[channel_id]['max'][i] = value
            else:
                self._metadata[channel_id]['max'][i] = max(self._metadata[channel_id]['max'][i], value)

        type_ = {
            spec.TYPE_UNSIGNED_BYTE: 'B',
            spec.TYPE_UNSIGNED_SHORT: 'H',
            spec.TYPE_UNSIGNED_INT: 'I',
            spec.TYPE_FLOAT: 'f',
        }[self._metadata[channel_id]['componentType']]

        format_ = '<{}'.format(type_ * size)

        self._channels[channel_id].write(struct.pack(format_, *values))
        self._metadata[channel_id]['count'] += 1

    def write_raw(self, channel_id, data):
        self._channels[channel_id].write(data)
        self._metadata[channel_id]['count'] += len(data)

    def count(self, channel_id):
        return self._metadata[channel_id]['count']

    def export(self, parent_node, filepath=None):
        offset = 0
        data = bytearray()
        uri = 'nothing.bin'

        # accessors + buffer views
        for i in range(len(self._channels)):
            channel = self._channels[i]
            metadata = self._metadata[i]
            extras = metadata.get('extras') or {}
            parent_node['accessors'].append(metadata)

            part = channel.getbuffer()
            view = {
                'buffer': len(parent_node['buffers']),
                'byteLength': len(part),
                'byteOffset': offset,
                'extras': extras,
            }
            parent_node['bufferViews'].append(view)

            offset += len(part)
            data += part

        # embedded images + buffer views
        for gltf_image in parent_node.get('images', []):
            extras = gltf_image.get('extras') or {}

            part = None

            if 'uri' in extras:
                tfilepath = os.path.join(os.path.dirname(self._filepath), extras['uri'])
                with open(tfilepath, 'rb') as f:
                    part = f.read()

            elif 'data' in extras:
                part = extras.pop('data')

            if not part:
                continue

            view = {
                'buffer': len(parent_node['buffers']),
                'byteLength': len(part),
                'byteOffset': offset,
                'extras': extras,
            }
            parent_node['bufferViews'].append(view)
            gltf_image['bufferView'] = len(parent_node['bufferViews']) - 1

            offset += len(part)
            data += part

        if filepath:
            buffer_fp = filepath.replace('.gltf', '.bin')
            with open(buffer_fp, 'wb') as f:
                f.write(data)

            uri = os.path.relpath(buffer_fp, os.path.dirname(self._filepath))

        if len(data):
            gltf_buffer = {
                'byteLength': len(data),
            }
            if filepath:
                gltf_buffer['uri'] = uri
            parent_node['buffers'].append(gltf_buffer)

        return data
