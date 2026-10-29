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

from panda3d.egg import EggMaterial

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
        egg_material = EggMaterial(material.name)

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
            return egg_material

        if self._render_type == 'rp':  # RenderPipeline
            # emission = tuple(shader.inputs['Emission'].default_value)[:3]
            emission = (0, 0, 0, 0)
            # alpha = shader.inputs['Alpha'].default_value
            alpha = 1
            clearcoat = shader.inputs['Clearcoat'].default_value
            normal_strength = self.get_normal_strength(material, shader)

            if sum(emission) > 0:  # emission
                egg_material.set_base(emission + (1, ))
                egg_material.set_metallic(0)
                egg_material.set_roughness(1)
                egg_material.set_ior(1.51)

                if alpha < 1:
                    emit = [SHADING_MODEL_TRANSPARENT_EMISSIVE, normal_strength, alpha, 0]
                else:
                    emit = [SHADING_MODEL_EMISSIVE, normal_strength, 0, 0]
                egg_material.set_emit(tuple(emit))

            else:  # not emission
                egg_material.set_base((1, 1, 1, 1))
                egg_material.set_metallic(self.get_metallic(material, shader))
                egg_material.set_roughness(self.get_roughness(material, shader))
                egg_material.set_ior(shader.inputs['IOR'].default_value)

                if alpha < 1:
                    emit = [SHADING_MODEL_TRANSPARENT_GLASS, normal_strength, alpha, 0]
                elif clearcoat:
                    emit = [SHADING_MODEL_CLEARCOAT, normal_strength, 0, 0]
                else:
                    emit = [SHADING_MODEL_DEFAULT, normal_strength, 0, 0]
                egg_material.set_emit(tuple(emit))

        else:  # not RenderPipeline
            emission = (0, 0, 0, 0)
            alpha = 1

            egg_material.set_base((1, 1, 1, alpha))
            egg_material.set_metallic(self.get_metallic(material, shader))
            egg_material.set_roughness(self.get_roughness(material, shader))
            egg_material.set_ior(shader.inputs['IOR'].default_value)
            egg_material.set_emit(emission)

        return egg_material
