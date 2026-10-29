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

import os

from . import spec


class TextureMixin(object):
    def get_images(self, material, shader):
        if self._render_type == 'rp':  # custom texture order for RP
            return (
                # p3d_Texture0 - baseColorTexture - RP Diffuse
                (('pbrMetallicRoughness', 'baseColorTexture'), self.get_diffuse(material, shader)),

                # p3d_Texture1 - metallicRoughnessTexture - RP Normal Map
                (('pbrMetallicRoughness', 'metallicRoughnessTexture'), self.get_normal_map(material, shader)),

                # p3d_Texture2 - normalTexture - RP IOR
                ('normalTexture', self.get_specular_map(material, shader)),

                # p3d_Texture3 - emissiveTexture - RP Roughness
                ('emissiveTexture', self.get_roughness_map(material, shader)),

                # put a placeholder, because all textures are required
                (None, ''),
            )
        else:
            return (
                (('pbrMetallicRoughness', 'baseColorTexture'), self.get_diffuse(material, shader)),
                (('pbrMetallicRoughness', 'metallicRoughnessTexture'), self.get_roughness_map(material, shader)),
                ('normalTexture', self.get_normal_map(material, shader)),
                ('emissiveTexture', self.get_emission_map(material, shader)),
                (None, ''),  # put a placeholder, because all textures are required
            )

    def make_texture(self, type_, image_texture):
        filepath = image_texture.image.filepath.lstrip('/')
        path = os.path.dirname(self._output)

        gltf_sampler = {
            'name': image_texture.image.name,
            'wrapS': spec.REPEAT,
            'wrapT': spec.REPEAT,
        }

        gltf_image = {
            'name': image_texture.image.name,
            'mimeType': 'image/{}'.format(image_texture.image.file_format.lower()),
        }

        if self._output.endswith('.gltf'):  # external texture
            gltf_image['uri'] = os.path.join(path, filepath)
        else:  # embedded texture
            gltf_image['extras'] = {}
            if image_texture.image.packed_file:
                gltf_image['extras']['data'] = image_texture.image.packed_file.data
            else:
                gltf_image['extras']['uri'] = os.path.join(self.get_cwd(), filepath)

        if image_texture.extension == 'CLIP':
            gltf_sampler['wrapS'] = spec.CLAMP_TO_EDGE
            gltf_sampler['wrapT'] = spec.CLAMP_TO_EDGE
        elif image_texture.extension == 'REPEAT':
            gltf_sampler['wrapS'] = spec.REPEAT
            gltf_sampler['wrapT'] = spec.REPEAT

        return gltf_sampler, gltf_image

    def make_empty_texture(self, type_):
        filepath = 'textures/unknown.png'

        if self._render_type == 'rp':
            if type(type_) == tuple:
                if 'baseColorTexture' in type_:
                    filepath = 'textures/diffuse.png'
                elif 'metallicRoughnessTexture' in type_:
                    filepath = 'textures/normal.png'
            elif type_ == 'normalTexture':
                filepath = 'textures/specular.png'
            elif type_ == 'emissiveTexture':
                filepath = 'textures/roughness.png'
        else:
            if type(type_) == tuple:
                if 'baseColorTexture' in type_:
                    filepath = 'textures/diffuse.png'
                elif 'metallicRoughnessTexture' in type_:
                    filepath = 'textures/metallic_roughness.png'
            elif type_ == 'normalTexture':
                filepath = 'textures/normal.png'
            elif type_ == 'emissiveTexture':
                filepath = 'textures/emissive.png'

        path = os.path.dirname(self._output)

        gltf_sampler = {
            'name': os.path.basename(filepath),
            'wrapS': spec.CLAMP_TO_EDGE,
            'wrapT': spec.CLAMP_TO_EDGE,
        }

        gltf_image = {
            'name': os.path.basename(filepath),
            'mimeType': 'image/png',
        }

        # gltf_image['uri'] = os.path.join(path, filepath)
        if self._output.endswith('.gltf'):  # external texture
            gltf_image['uri'] = os.path.join(path, filepath)
        else:  # embedded texture
            gltf_image['extras'] = {
                'uri': os.path.join(self.get_cwd(), filepath),
            }

        return gltf_sampler, gltf_image
