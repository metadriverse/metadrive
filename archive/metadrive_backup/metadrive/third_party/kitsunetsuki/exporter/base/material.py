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

from metadrive.third_party.kitsunetsuki.base.material import get_from_node


class MaterialMixin(object):
    def get_metallic(self, material, shader):
        if shader.type in ('BSDF_GLASS', 'BSDF_ANISOTROPIC'):
            return 1

        # Math [Value] -> [Metallic] Principled BSDF
        # math_node = get_from_node(
        #     material.node_tree, 'MATH', to_node=shader,
        #     from_socket_name='Value', to_socket_name='Metallic')
        # if math_node:
        #     for input_ in math_node.inputs:
        #         if input_.name == 'Value' and not input_.is_linked:
        #             if input_.default_value < 0.5:
        #                 return 0
        #             else:
        #                 return 1

        # elif not shader.inputs['Metallic'].is_linked:
        #     return shader.inputs['Metallic'].default_value

        if shader.inputs['Metallic'].is_linked:
            return 0  # metallic map is not supported in RP -> disable
        else:
            return shader.inputs['Metallic'].default_value

    def get_roughness(self, material, shader):
        # Math [Value] -> [Roughness] Principled BSDF
        math_node = get_from_node(
            material.node_tree, 'MATH', to_node=shader, from_socket_name='Value', to_socket_name='Roughness'
        )
        if math_node:
            for input_ in math_node.inputs:
                if input_.name == 'Value' and not input_.is_linked:
                    return input_.default_value
        else:
            return shader.inputs['Roughness'].default_value

    def get_emission(self, material, shader):
        # if not shader.inputs['Emission'].is_linked and not shader.inputs['Emission Strength'].is_linked:
        #     r, g, b, *_ = shader.inputs['Emission'].default_value
        #     e = shader.inputs['Emission Strength'].default_value
        #     return (r * e, g * e, b * e)

        # Mix RGB [Color] -> [Emission] Principled BSDF
        mix_node = get_from_node(
            material.node_tree, 'MIX_RGB', to_node=shader, from_socket_name='Color', to_socket_name='Emission'
        )
        if mix_node:
            for input_ in mix_node.inputs:
                if input_.name.startswith('Color') and not input_.is_linked:
                    return input_.default_value
        else:
            return shader.inputs['Emission'].default_value

    def get_normal_strength(self, material, shader):
        # Normal Map [Normal] -> [Normal] Principled BSDF
        normal_map = get_from_node(
            material.node_tree, 'NORMAL_MAP', to_node=shader, from_socket_name='Normal', to_socket_name='Normal'
        )
        if normal_map:
            return normal_map.inputs['Strength'].default_value
        else:
            return 1
