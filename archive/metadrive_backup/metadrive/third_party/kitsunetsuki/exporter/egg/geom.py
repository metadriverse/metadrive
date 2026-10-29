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

from panda3d.egg import (EggGroup, EggPolygon, EggVertexPool, EggMaterial, EggTexture)

from metadrive.third_party.kitsunetsuki.base.matrices import get_object_matrix
from metadrive.third_party.kitsunetsuki.base.armature import get_armature
from metadrive.third_party.kitsunetsuki.base.mesh import obj2mesh
from metadrive.third_party.kitsunetsuki.base.objects import apply_modifiers, is_collision


class GeomMixin(object):
    def _get_joints(self, egg_group):
        results = {}

        for child in egg_group.get_children():
            if (isinstance(child, EggGroup) and child.get_group_type() == EggGroup.GT_joint):
                results[child.get_name()] = child
                results.update(self._get_joints(child))

        return results

    def make_geom(self, node, obj, can_merge=False):
        triangulate = not is_collision(obj)
        if self._geom_scale != 1:
            obj.scale.x = self._geom_scale
            obj.scale.y = self._geom_scale
            obj.scale.z = self._geom_scale
            apply_modifiers(obj, triangulate=triangulate, apply_scale=True)
        else:
            apply_modifiers(obj, triangulate=triangulate)
        mesh = obj2mesh(obj, triangulate=triangulate)

        # get or create materials and textures
        egg_materials = {}
        egg_textures = {}
        egg_material_textures = {}
        if not self._no_materials and not is_collision(obj):
            for material in mesh.materials.values():
                # material
                for child in self._root.get_children():  # existing material
                    if (isinstance(child, EggMaterial) and child.get_name() == material.name):
                        egg_materials[material.name] = child
                        break
                else:  # new material
                    egg_material = self.make_material(material)
                    self._root.add_child(egg_material)
                    egg_materials[material.name] = egg_material

                # material -> textures
                if material.name not in egg_material_textures:
                    egg_material_textures[material.name] = {}

                # textures
                if not self._no_textures:
                    for type_, _, egg_texture in self.make_textures(material):
                        tname = egg_texture.get_name()
                        for child in self._root.get_children():  # existing texture
                            if (isinstance(child, EggTexture) and child.get_name() == tname):
                                egg_textures[tname] = child
                                egg_material_textures[material.name][tname] = child
                                break
                        else:  # new texture
                            self._root.add_child(egg_texture)
                            egg_textures[tname] = egg_texture
                            egg_material_textures[material.name][tname] = egg_texture

        # get or create vertex pool
        egg_vertex_pool = None
        egg_vertex_id = 0
        if can_merge:
            for child in node.get_children():  # existing vertex pool
                if isinstance(child, EggVertexPool):
                    egg_vertex_pool = child
                    egg_vertex_id = egg_vertex_pool.get_highest_index()
                    break

        if egg_vertex_pool is None:  # new vertex pool
            egg_vertex_pool = EggVertexPool(node.get_name())
            egg_vertex_id = egg_vertex_pool.get_highest_index()
            node.add_child(egg_vertex_pool)

        # get armature and joints
        armature = get_armature(obj)
        egg_joints = {}
        if armature:
            for child in self._root.get_children():
                if (isinstance(child, EggGroup) and child.get_dart_type() == EggGroup.DT_structured
                        and child.get_name() == armature.name):
                    egg_joints = self._get_joints(child)

        sharp_vertices = {}
        uv_tb = {}
        if not is_collision(obj):
            sharp_vertices = self.get_sharp_vertices(mesh)
            uv_tb = self.get_tangent_bitangent(mesh)
        egg_vertices = {}

        obj_matrix = get_object_matrix(obj, armature)
        parent_obj_matrix = obj_matrix
        if armature:
            parent_obj_matrix = get_object_matrix(armature)

        for polygon in mesh.polygons:
            # <-- polygon
            material = None
            mname = None
            if not self._no_materials:
                try:
                    material = mesh.materials[polygon.material_index]
                    mname = material.name
                except IndexError:
                    pass

            # make polygon
            egg_polygon = EggPolygon(mname or node.get_name())

            # set material and textures
            if material and not self._no_materials and not is_collision(obj):
                if mname in egg_materials:
                    egg_polygon.set_material(egg_materials[mname])

                # set textures
                if mname in egg_material_textures and not self._no_textures:
                    for egg_texture in egg_material_textures[mname].values():
                        egg_polygon.add_texture(egg_texture)

            # vertices
            for i, vertex_id in enumerate(polygon.vertices):
                # i is vertex counter inside a polygon
                # (0, 1, 2) for triangle
                # vertex_id is reusable id,
                # because multiple polygons can share the same vertices

                # <-- vertex
                vertex = mesh.vertices[vertex_id]
                use_smooth = (polygon.use_smooth and vertex_id not in sharp_vertices and not is_collision(obj))

                # try to reuse shared vertices
                if (polygon.use_smooth and vertex_id in egg_vertices and not is_collision(obj)):
                    shared = False
                    for egg_vertex in egg_vertices[vertex_id]:
                        loop_id = polygon.loop_indices[i]
                        egg_vertex_uv = egg_vertex.get_uv_obj(self._get_uv_name(mesh.uv_layers.active))

                        if not egg_vertex_uv:
                            egg_polygon.add_vertex(egg_vertex)
                            shared = True
                            break

                        if self.can_share_vertex(mesh, vertex, loop_id, egg_vertex_uv.get_uv(),
                                                 egg_vertex.get_normal()):
                            egg_polygon.add_vertex(egg_vertex)
                            shared = True
                            break
                    if shared:
                        continue

                # make new vertex data
                can_merge_vertices = can_merge
                if armature:
                    can_merge_vertices = True
                elif is_collision(obj):
                    can_merge_vertices = False
                egg_vertex = self.make_vertex(
                    parent_obj_matrix, obj_matrix, polygon, vertex, use_smooth=use_smooth, can_merge=can_merge_vertices
                )

                # uv layers
                if not is_collision(obj):
                    for uv_name, uv_layer in mesh.uv_layers.items():
                        # <-- vertex uv
                        loop_id = polygon.loop_indices[i]
                        uv_loop = uv_layer.data[loop_id]

                        # not active layer and extra UV disabled
                        if not uv_layer.active and self._no_extra_uv:
                            continue

                        egg_vertex_uv = self.make_vertex_uv(uv_layer, uv_loop.uv)
                        if uv_name in uv_tb:
                            t, b, s = uv_tb[uv_name][loop_id]
                            tangent = parent_obj_matrix @ t
                            binormal = parent_obj_matrix @ b
                            egg_vertex_uv.set_tangent(tuple(tangent))
                            egg_vertex_uv.set_binormal(tuple(binormal))
                        egg_vertex.set_uv_obj(egg_vertex_uv)
                        # vertex uv -->

                # generate new ID, add vertex and save last ID
                egg_vertex_id += 1
                egg_vertex_pool.add_vertex(egg_vertex, egg_vertex_id)
                egg_vertex_pool.set_highest_index(egg_vertex_id)
                egg_polygon.add_vertex(egg_vertex)

                # save vertex data for sharing
                if vertex_id not in egg_vertices:
                    egg_vertices[vertex_id] = []
                egg_vertices[vertex_id].append(egg_vertex)

                # attach joints to vertex
                if armature:
                    for vertex_group in vertex.groups:
                        obj_vertex_group = obj.vertex_groups[vertex_group.group]
                        if obj_vertex_group.name in egg_joints:
                            egg_joint = egg_joints[obj_vertex_group.name]
                            egg_joint.set_vertex_membership(egg_vertex, vertex_group.weight)

                # vertex -->

            node.add_child(egg_polygon)
            # polygon -->
