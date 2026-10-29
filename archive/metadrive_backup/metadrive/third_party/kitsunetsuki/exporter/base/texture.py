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

from metadrive.third_party.kitsunetsuki.base.material import get_root_node, get_from_node


class TextureMixin(object):
    def get_diffuse(self, material, shader):
        for i in ('Color', 'Alpha'):
            # Image Texture [Color/Alpha] -> [Socket] Principled BSDF
            node = get_from_node(
                material.node_tree, 'TEX_IMAGE', to_node=shader, from_socket_name=i, to_socket_name='Base Color'
            )
            if node:
                return node

    def get_normal_map(self, material, shader):
        # Normal Map [Normal] -> [Normal] Principled BSDF
        normal_map = get_from_node(
            material.node_tree, 'NORMAL_MAP', to_node=shader, from_socket_name='Normal', to_socket_name='Normal'
        )
        if normal_map:
            # Image Texture [Color] -> [Color] Normal Map
            return get_from_node(
                material.node_tree, 'TEX_IMAGE', to_node=normal_map, from_socket_name='Color', to_socket_name='Color'
            )

    def get_emission_map(self, material, shader):
        # emission map pipeline
        # Image Texture [Color] -> [Emission Strength] Principled BSDF
        if shader.inputs['Emission Strength'].is_linked:
            node = get_from_node(
                material.node_tree,
                'TEX_IMAGE',
                to_node=shader,
                from_socket_name='Color',
                to_socket_name='Emission Strength'
            )
            if node:
                return node

        # emission color pipeline
        # Mix RGB [Color] -> [Emission] Principled BSDF
        mix_node = get_from_node(
            material.node_tree, 'MIX_RGB', to_node=shader, from_socket_name='Color', to_socket_name='Emission'
        )
        if mix_node:
            # Image Texture [Color] -> [Color1/Color2] Mix
            for input_ in mix_node.inputs:
                if input_.name.startswith('Color') and input_.is_linked:
                    return get_from_node(
                        material.node_tree,
                        'TEX_IMAGE',
                        to_node=mix_node,
                        from_socket_name='Color',
                        to_socket_name=input_.name
                    )
        else:
            # Image Texture [Color] -> [Emission] Principled BSDF
            return get_from_node(
                material.node_tree, 'TEX_IMAGE', to_node=shader, from_socket_name='Color', to_socket_name='Emission'
            )

    def get_specular_map(self, material, shader):
        # Math [Value] -> [Specular] Principled BSDF
        math_node = get_from_node(
            material.node_tree, 'MATH', to_node=shader, from_socket_name='Value', to_socket_name='Specular'
        )
        if math_node:
            # Image Texture [Color] -> [Input] Math
            return get_from_node(
                material.node_tree, 'TEX_IMAGE', to_node=math_node, from_socket_name='Color', to_socket_name='Value'
            )
        else:
            # Image Texture [Color] -> [Specular] Principled BSDF
            return get_from_node(
                material.node_tree, 'TEX_IMAGE', to_node=shader, from_socket_name='Color', to_socket_name='Specular'
            )

    def get_roughness_map(self, material, shader):
        # Math [Value] -> [Roughness] Principled BSDF
        math_node = get_from_node(
            material.node_tree, 'MATH', to_node=shader, from_socket_name='Value', to_socket_name='Roughness'
        )
        if math_node:
            # Image Texture [Color] -> [Input] Math
            return get_from_node(
                material.node_tree, 'TEX_IMAGE', to_node=math_node, from_socket_name='Color', to_socket_name='Value'
            )
        else:
            # Image Texture [Color] -> [Roughness] Principled BSDF
            return get_from_node(
                material.node_tree, 'TEX_IMAGE', to_node=shader, from_socket_name='Color', to_socket_name='Roughness'
            )

    def get_parallax_map(self, material, shader):
        return

    def make_texture(self, i, image_texture):
        raise NotImplementedError()

    def make_empty_texture(self, i):
        raise NotImplementedError()

    def get_images(self, material, shader):
        return tuple()

    def make_textures(self, material):
        results = []

        shader = None
        if material.node_tree is not None:
            output = get_root_node(material.node_tree, 'OUTPUT_MATERIAL')
            if output:
                shader = get_from_node(
                    material.node_tree,
                    'BSDF_PRINCIPLED',
                    to_node=output,
                    from_socket_name='BSDF',
                    to_socket_name='Surface'
                )

        if shader:
            image_textures = self.get_images(material, shader)
            last_texid = 0
            for i, (type_, image_texture) in enumerate(reversed(image_textures)):
                if image_texture is not None:
                    last_texid = len(image_textures) - i - 1
                    break

            for i, (type_, image_texture) in enumerate(image_textures):
                if image_texture is None:
                    if self._empty_textures:  # fill empty slot
                        result = self.make_empty_texture(type_)
                        results.append((type_, ) + result)
                    elif self._render_type == 'rp':
                        break

                elif image_texture:
                    result = self.make_texture(type_, image_texture)
                    results.append((type_, ) + result)

                if i >= last_texid and self._render_type == 'rp':
                    break

        return results
