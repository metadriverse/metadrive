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

from metadrive.third_party.kitsunetsuki.base.armature import get_armature
from metadrive.third_party.kitsunetsuki.base.matrices import get_object_matrix
from metadrive.third_party.kitsunetsuki.base.mesh import obj2mesh
from metadrive.third_party.kitsunetsuki.base.objects import apply_modifiers, is_collision

from . import spec


class GeomMixin(object):
    def _get_joints(self, gltf_node):
        results = {}

        for i, child_id in enumerate(gltf_node['joints']):
            child = self._root['nodes'][child_id]
            results[child['name']] = i

        return results

    def _make_primitive(self, gltf_mesh, mesh):
        gltf_primitive = {
            'attributes': {},
            'material': 0,
            # 'mode': 4,
            'extras': {
                'highest_index': -1,
                'targetNames': gltf_mesh['extras']['targetNames'],
            },
        }

        channel = self._buffer.add_channel(
            {
                # 'componentType': spec.TYPE_UNSIGNED_SHORT,
                'componentType': spec.TYPE_UNSIGNED_INT,
                'type': 'SCALAR',
                'extras': {
                    'reference': 'indices',
                },
            }
        )
        gltf_primitive['indices'] = channel['bufferView']

        if gltf_mesh['primitives'] and not self._split_primitives:
            gltf_primitive['attributes'] = gltf_mesh['primitives'][0]['attributes']
            if 'targets' in gltf_mesh['primitives'][0]:
                gltf_primitive['targets'] = gltf_mesh['primitives'][0]['targets']

        else:
            channel = self._buffer.add_channel(
                {
                    'componentType': spec.TYPE_FLOAT,
                    'type': 'VEC3',
                    'extras': {
                        'reference': 'NORMAL',
                    },
                }
            )
            gltf_primitive['attributes']['NORMAL'] = channel['bufferView']

            channel = self._buffer.add_channel(
                {
                    'componentType': spec.TYPE_FLOAT,
                    'type': 'VEC3',
                    'extras': {
                        'reference': 'POSITION',
                    },
                }
            )
            gltf_primitive['attributes']['POSITION'] = channel['bufferView']

            for sk_id, sk_name in enumerate(gltf_primitive['extras']['targetNames']):
                gltf_target = {
                    # '_extras': {
                    #     'name': sk_name,
                    # },
                }

                channel = self._buffer.add_channel(
                    {
                        'componentType': spec.TYPE_FLOAT,
                        'type': 'VEC3',
                        'extras': {
                            'reference': 'POSITION',
                            'target': sk_name,
                        },
                    }
                )
                gltf_target['POSITION'] = channel['bufferView']

                if 'targets' not in gltf_primitive:
                    gltf_primitive['targets'] = []
                gltf_primitive['targets'].append(gltf_target)

        return gltf_primitive

    def make_geom(self, gltf_node, gltf_mesh, obj, can_merge=False):
        triangulate = True
        if self._geom_scale != 1:
            scale = obj.scale
            obj.scale.x, obj.scale.y, obj.scale.z = [self._geom_scale] * 3
            apply_modifiers(obj, triangulate=triangulate, apply_scale=True)
            obj.scale = scale
        else:
            apply_modifiers(obj, triangulate=triangulate)
        mesh = obj2mesh(obj, triangulate=triangulate)

        # setup shape key names for the primitives
        if mesh.shape_keys:
            for sk_name in sorted(mesh.shape_keys.key_blocks.keys()):
                if sk_name.lower() == 'basis':
                    continue

                gltf_mesh['extras']['targetNames'].append(sk_name)

        # get or create materials and textures
        gltf_materials = {}
        if not self._no_materials and not is_collision(obj):
            for material in mesh.materials.values():
                # empty material slot
                if not material:
                    continue

                # material
                for i, child in enumerate(self._root['materials']):  # existing material
                    if child['name'] == material.name:
                        gltf_materials[material.name] = i
                        break
                else:  # new material
                    gltf_material = self.make_material(material)
                    self._root['materials'].append(gltf_material)

                    gltf_materials[material.name] = len(self._root['materials']) - 1

                # textures
                if not self._no_textures:
                    for type_, gltf_sampler, gltf_image in self.make_textures(material):
                        tname = gltf_image['name']
                        for i, child in enumerate(self._root['images']):  # existing texture
                            if child['name'] == tname:
                                texid = i
                                break
                        else:  # new texture
                            self._root['samplers'].append(gltf_sampler)
                            self._root['images'].append(gltf_image)

                            gltf_texture = {
                                'sampler': len(self._root['samplers']) - 1,
                                'source': len(self._root['images']) - 1,
                            }
                            self._root['textures'].append(gltf_texture)
                            texid = len(self._root['textures']) - 1

                        matid = gltf_materials[material.name]
                        if type(type_) == tuple and len(type_) == 2:
                            type_l1, type_l2 = type_
                            self._root['materials'][matid][type_l1][type_l2] = {
                                'index': texid,
                                'texCoord': 0,
                            }
                        else:
                            self._root['materials'][matid][type_] = {
                                'index': texid,
                                'texCoord': 0,
                            }

        # get primitives
        gltf_primitives = {}
        gltf_primitive_indices = {}  # splitted vertex buffers
        gltf_mesh_vertices_index = -1  # reusable vertex buffer
        if can_merge:
            for i, gltf_primitive in enumerate(gltf_mesh['primitives']):
                mname = None
                if 'material' in gltf_primitive:
                    matid = gltf_primitive['material']
                    mname = self._root['materials'][matid]['name']
                gltf_primitives[mname] = gltf_primitive
                gltf_primitive_indices[mname] = gltf_primitive['extras']['highest_index']

        gltf_vertices = {}

        # get armature and joints
        armature = get_armature(obj)
        max_joints = 0  # get max joints per vertex
        gltf_joints = {}
        if armature:
            # max_joints = 1
            # for polygon in mesh.polygons:
            #     for vertex_id in polygon.vertices:
            #         vertex = mesh.vertices[vertex_id]
            #         joints = 0
            #         for vertex_group in vertex.groups:
            #             obj_vertex_group = obj.vertex_groups[vertex_group.group]
            #             if vertex_group.weight > 0:
            #                 joints += 1
            #         max_joints = max(max_joints, joints)

            if 'skin' in gltf_node:
                gltf_skin = self._root['skins'][gltf_node['skin']]

                # for i, child in enumerate(self._root['skins']):
                #     if child['name'] == armature.name:
                #         gltf_joints = self._get_joints(child)
                #         break
                gltf_joints = self._get_joints(gltf_skin)

        # get max joint layers (4 bones per layer)
        # max_joint_layers = math.ceil(max_joints / 4)

        # panda3d-gltf is limited to 1 single layer only (up to 4 bones)
        max_joint_layers = 1

        sharp_vertices = self.get_sharp_vertices(mesh)
        uv_tb = self.get_tangent_bitangent(mesh)
        obj_matrix = self._transform(get_object_matrix(obj, armature=armature))

        for polygon in mesh.polygons:
            # <-- polygon
            material = None
            mname = None
            if not self._no_materials:
                try:
                    material = mesh.materials[polygon.material_index]
                    if material:
                        mname = material.name
                except IndexError:
                    pass

            # get or create primitive
            if mname in gltf_primitives:
                gltf_primitive = gltf_primitives[mname]
            else:
                gltf_primitive = self._make_primitive(gltf_mesh, mesh)
                gltf_primitives[mname] = gltf_primitive
                gltf_primitive_indices[mname] = -1
                gltf_mesh['primitives'].append(gltf_primitive)

            # set material
            if material and not self._no_materials and not is_collision(obj):
                if material.name in gltf_materials:
                    gltf_primitive['material'] = gltf_materials[mname]

            # vertices
            for i, vertex_id in enumerate(polygon.vertices):
                # i is vertex counter inside a polygon
                # (0, 1, 2) for triangle
                # vertex_id is reusable id,
                # because multiple polygons can share the same vertices

                loop_id = polygon.loop_indices[i]

                # <-- vertex
                vertex = mesh.vertices[vertex_id]
                use_smooth = (
                    polygon.use_smooth and
                    # vertex_id not in sharp_vertices and
                    not is_collision(obj)
                )

                # try to reuse shared vertices
                if mname not in gltf_vertices:
                    gltf_vertices[mname] = {}
                if (polygon.use_smooth and vertex_id in gltf_vertices[mname] and not is_collision(obj)):
                    shared = False
                    for gltf_vertex_index, gltf_vertex_uv, gltf_vertex_normal in gltf_vertices[mname][vertex_id]:
                        if self.can_share_vertex(mesh, vertex, loop_id, gltf_vertex_uv, gltf_vertex_normal):
                            self._buffer.write(gltf_primitive['indices'], gltf_vertex_index)
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
                self.make_vertex(
                    obj_matrix,
                    gltf_primitive,
                    mesh,
                    polygon,
                    vertex,
                    vertex_id,
                    loop_id,
                    use_smooth=use_smooth,
                    can_merge=can_merge_vertices
                )

                # uv layers, active first
                active_uv = 0, 0
                if not is_collision(obj):
                    uv_layers = sorted(mesh.uv_layers.items(), key=lambda x: not x[1].active)
                    for uv_id, (uv_name, uv_layer) in enumerate(uv_layers):
                        # <-- vertex uv
                        uv_loop = uv_layer.data[loop_id]

                        # not active layer and extra UV disabled
                        if not uv_layer.active and self._no_extra_uv:
                            continue

                        u, v = uv_loop.uv.to_2d()
                        if uv_layer.active:
                            active_uv = u, v
                        self._write_uv(gltf_primitive, uv_id, u, v)
                        if uv_name in uv_tb and uv_layer.active:
                            self._write_tbs(
                                obj_matrix, gltf_primitive, *uv_tb[uv_name][loop_id], can_merge=can_merge_vertices
                            )
                        # vertex uv -->

                # generate new ID, add vertex and save last ID
                gltf_primitive_indices[mname] += 1
                gltf_mesh_vertices_index += 1
                if self._split_primitives:
                    idx = gltf_primitive_indices[mname]
                else:
                    idx = gltf_mesh_vertices_index
                self._buffer.write(gltf_primitive['indices'], idx)
                gltf_primitive['extras']['highest_index'] = gltf_primitive_indices[mname]

                # save vertex data for sharing
                if vertex_id not in gltf_vertices[mname]:
                    gltf_vertices[mname][vertex_id] = []
                gltf_vertices[mname][vertex_id].append(
                    (
                        idx,
                        active_uv,
                        mesh.loops[loop_id].normal if use_smooth else polygon.normal,
                    )
                )

                # attach joints to vertex
                if gltf_joints:
                    joints_weights = []
                    vertex_groups = reversed(sorted(vertex.groups, key=lambda vg: vg.weight))
                    for vertex_group in vertex_groups:
                        obj_vertex_group = obj.vertex_groups[vertex_group.group]

                        # no bones with vertex group's name
                        if obj_vertex_group.name not in gltf_joints:
                            continue

                        # weight is zero
                        if vertex_group.weight <= 0:
                            continue

                        joint_id = gltf_joints[obj_vertex_group.name]
                        joints_weights.append([joint_id, vertex_group.weight])

                    # objects reparented to bone instead of entire armature
                    if obj.parent_type == 'BONE' and obj.parent_bone in gltf_joints:
                        joint_id = gltf_joints[obj.parent_bone]
                        joints_weights.append([joint_id, 1])

                    # padding
                    while ((len(joints_weights) % 4 != 0) or (len(joints_weights) < max_joint_layers * 4)):
                        joints_weights.append([0, 0])

                    # limit by max joints
                    joints_weights = joints_weights[:max_joint_layers * 4]

                    imax = -1
                    wmax = 0
                    for j, (joint, weight) in enumerate(joints_weights):
                        if weight > wmax:
                            imax = j
                            wmax = weight
                    # if self._norm_weights and imax >= 0:
                    #     joints_weights[imax][1] += 1 - sum(list(zip(*joints_weights))[1])
                    if self._norm_weights:
                        weights = list(zip(*joints_weights))[1]
                        weights_sum = sum(weights)
                        for j in range(len(joints_weights)):
                            joints_weights[j][1] *= 1 / weights_sum

                    # group by 4 joint-weight pairs
                    joints_weights_groups = []
                    for j in range(len(joints_weights) // 4):
                        group = joints_weights[j * 4:j * 4 + 4]
                        joints_weights_groups.append(group)

                    self._write_joints_weights(gltf_primitive, len(tuple(gltf_joints.keys())), joints_weights_groups)

                # vertex -->
            # polygon -->
