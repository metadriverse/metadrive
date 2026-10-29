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

SHADING_MODEL_DEFAULT = 0
SHADING_MODEL_EMISSIVE = 1
SHADING_MODEL_CLEARCOAT = 2
SHADING_MODEL_TRANSPARENT_GLASS = 3
SHADING_MODEL_SKIN = 4
SHADING_MODEL_FOLIAGE = 5
SHADING_MODEL_TRANSPARENT_EMISSIVE = 6


class MaterialMixin(object):
    def make_material(self, material):
        gltf_material = {
            'name': material.name,
            'alphaMode': 'OPAQUE',
            'alphaCutoff': material.alpha_threshold,
            'doubleSided': not material.use_backface_culling,
            'pbrMetallicRoughness': {
                'extras': {},
            },
            'extras': {},
        }

        gltf_material['alphaMode'] = {
            'OPAQUE': 'OPAQUE',
            'BLEND': 'BLEND',
            'CLIP': 'MASK'
        }.get(material.blend_method, 'OPAQUE')

        shader = None
        if material.node_tree is not None:
            output = get_root_node(material.node_tree, 'OUTPUT_MATERIAL')
            shader = None
            if output:
                shader = get_from_node(
                    material.node_tree,
                    'BSDF_PRINCIPLED',
                    to_node=output,
                    from_socket_name='BSDF',
                    to_socket_name='Surface'
                )

        if not shader:
            return gltf_material

        if self._render_type == 'rp':  # RenderPipeline
            # emission = tuple(shader.inputs['Emission'].default_value)[:3]
            emission = (0, 0, 0)
            # alpha = shader.inputs['Alpha'].default_value
            alpha = 1
            clearcoat = shader.inputs['Clearcoat'].default_value
            normal_strength = self.get_normal_strength(material, shader)

            if sum(emission) > 0:  # emission
                gltf_material['pbrMetallicRoughness'].update(
                    {
                        'baseColorFactor': emission + (1, ),
                        'metallicFactor': 0,
                        'roughnessFactor': 1,
                    }
                )
                gltf_material['pbrMetallicRoughness']['extras']['ior'] = 1.51

                if alpha < 1:
                    emit = [SHADING_MODEL_TRANSPARENT_EMISSIVE, normal_strength, alpha]
                else:
                    emit = [SHADING_MODEL_EMISSIVE, normal_strength, 0]
                gltf_material['emissiveFactor'] = tuple(emit)

            else:  # not emission
                gltf_material['pbrMetallicRoughness'].update(
                    {
                        'baseColorFactor': (1, 1, 1, 1),
                        'metallicFactor': self.get_metallic(material, shader),
                        'roughnessFactor': self.get_roughness(material, shader),
                    }
                )
                gltf_material['pbrMetallicRoughness']['extras']['ior'] = shader.inputs['IOR'].default_value

                if alpha < 1:
                    emit = [SHADING_MODEL_TRANSPARENT_GLASS, normal_strength, alpha]
                elif clearcoat:
                    emit = [SHADING_MODEL_CLEARCOAT, normal_strength, 0]
                else:
                    emit = [SHADING_MODEL_DEFAULT, normal_strength, 0]
                gltf_material['emissiveFactor'] = tuple(emit)

        else:  # not RenderPipeline
            alpha = 1

            gltf_material['pbrMetallicRoughness'].update(
                {
                    'baseColorFactor': (1, 1, 1, alpha),
                    'metallicFactor': self.get_metallic(material, shader),
                    'roughnessFactor': self.get_roughness(material, shader),
                }
            )

            gltf_material['emissiveFactor'] = tuple(self.get_emission(material, shader))[:3]
            # if alpha < 1:
            #     gltf_material['alphaMode'] = 'BLEND'
            #     gltf_material['alphaCutoff'] = alpha
            # else:
            #     gltf_material['alphaMode'] = 'OPAQUE'
            #     gltf_material['alphaCutoff'] = 0

        if material.node_tree is not None:
            for node in material.node_tree.nodes:
                if node.type == 'ATTRIBUTE':
                    gltf_material['extras'][node.attribute_type.lower()] = node.attribute_name

        return gltf_material
