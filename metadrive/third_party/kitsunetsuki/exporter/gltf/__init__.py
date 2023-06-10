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

import bpy
import json
import math
import mathutils  # make sure to "import bpy" before
import struct

from bpy_extras.io_utils import ExportHelper
from typing import Set, cast

from metadrive.third_party.kitsunetsuki.base.armature import get_armature
from metadrive.third_party.kitsunetsuki.base.context import Mode
from metadrive.third_party.kitsunetsuki.base.collections import get_object_collection
from metadrive.third_party.kitsunetsuki.base.matrices import (
    get_bone_matrix, get_object_matrix, get_inverse_bind_matrix, matrix_to_list, quat_to_list
)
from metadrive.third_party.kitsunetsuki.base.objects import (
    is_collision, is_object_visible, get_object_properties, set_active_object
)

from metadrive.third_party.kitsunetsuki.exporter.base import Exporter

from . import spec
from .buffer import GLTFBuffer
from .animation import AnimationMixin
from .geom import GeomMixin
from .material import MaterialMixin
from .vertex import VertexMixin
from .texture import TextureMixin


class GLTFExporter(AnimationMixin, GeomMixin, MaterialMixin, VertexMixin, TextureMixin, Exporter):
    """
    BLEND to GLTF converter.
    """
    def __init__(self, args):
        super().__init__(args)
        self._output = args.output or args.inputs[0].replace('.blend', '.gltf')
        self._z_up = getattr(args, 'z_up', False)
        self._pose_freeze = getattr(args, 'pose_freeze', False)
        self._split_primitives = getattr(args, 'split_primitives', False)
        self._norm_weights = getattr(args, 'normalize_weights', False)

        if self._z_up:
            self._matrix = mathutils.Matrix((
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ))
            self._matrix_inv = self._matrix
        else:
            # Mat3.convert_mat(CS_yup_right, CS_zup_right)
            self._matrix = mathutils.Matrix((
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, -1.0, 0.0),
            ))

            # Mat3.convert_mat(CS_zup_right, CS_yup_right)
            self._matrix_inv = mathutils.Matrix((
                (1.0, 0.0, 0.0),
                (0.0, 0.0, -1.0),
                (0.0, 1.0, 0.0),
            ))

    def _transform(self, x):
        """
        Transform matrix/vector using axis conversion matrices.
        """
        if self._z_up:
            return x
        else:
            return (self._matrix.to_4x4() @ x @ self._matrix_inv.to_4x4())

    def _freeze(self, matrix):
        """
        Freezes matrix. Removes rotation and scale.
        """
        pos = matrix.to_translation()
        return mathutils.Matrix.Translation(pos).to_4x4()

    def make_root_node(self):
        gltf_node = {
            'asset': {
                'generator': (
                    'KITSUNETSUKI Asset Tools by kitsune.ONE - '
                    'https://github.com/kitsune-ONE-team/KITSUNETSUKI-Asset-Tools'
                ),
                'version': '2.0',
            },
            'extensions': {
                # 'BP_physics_engine': {'engine': 'bullet'},
            },
            'extensionsUsed': [],
            'scene': 0,
            'scenes': [{
                'name': 'Scene',
                'nodes': [],
            }],
            'nodes': [],
            'meshes': [],
            'materials': [{
                # skips panda warnings
                'name': 'GLTF_DEFAULT_MATERIAL',
            }],
            'animations': [],
            'skins': [],
            'textures': [],  # links to samplers-images pair
            'samplers': [],
            'images': [],
            'accessors': [],
            'bufferViews': [],
            'buffers': [],
        }

        if self._z_up:
            gltf_node['extensionsUsed'].append('BP_zup')

        return gltf_node

    def _add_child(self, parent_node, child_node):
        self._root['nodes'].append(child_node)
        node_id = len(self._root['nodes']) - 1

        if 'scenes' in parent_node:
            self._root['scenes'][0]['nodes'].append(node_id)
        else:
            if 'children' not in parent_node:
                parent_node['children'] = []
            parent_node['children'].append(node_id)

    def _setup_node(self, node, obj=None, can_merge=False):
        if obj is None:
            return

        armature = get_armature(obj)
        obj_matrix = self._transform(get_object_matrix(obj, armature=armature))

        # get custom object properties
        obj_props = get_object_properties(obj)

        if not can_merge and not armature:
            if self._geom_scale == 1:
                node.update(
                    {
                        'rotation': quat_to_list(obj_matrix.to_quaternion()),
                        'scale': list(obj_matrix.to_scale()),
                        'translation': list(obj_matrix.to_translation()),
                        # 'matrix': matrix_to_list(obj_matrix),
                    }
                )
            else:
                x, y, z = list(obj_matrix.to_translation())
                x *= self._geom_scale
                y *= self._geom_scale
                z *= self._geom_scale
                node.update(
                    {
                        'rotation': quat_to_list(obj_matrix.to_quaternion()),
                        'scale': list(obj_matrix.to_scale()),
                        'translation': [x, y, z],
                    }
                )

        # setup collisions
        if not can_merge and is_collision(obj) and obj_props.get('type') != 'Portal':
            collision = {}
            node['extensions'] = {
                'BLENDER_physics': collision,
            }

            # collision shape
            shape = {
                'shapeType': obj.rigid_body.collision_shape,
                'boundingBox': [obj.dimensions[i] / obj_matrix.to_scale()[i] * self._geom_scale for i in range(3)],
            }
            if obj.rigid_body.collision_shape == 'MESH':
                shape['mesh'] = node.pop('mesh')

            collision['collisionShapes'] = [shape]
            collision['static'] = obj.rigid_body.type == 'PASSIVE'

            # don't actually collide (ghost)
            if (not obj.collision or not obj.collision.use):
                collision['intangible'] = True

            if self._set_origin:
                obj.select_set(state=True)
                set_active_object(obj)
                x1, y1, z1 = obj.location
                # set origin to the center of bounds
                bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
                x2, y2, z2 = obj.location
                if 'extras' not in node:
                    node['extras'] = {}
                node['extras']['origin'] = [x2 - x1, y2 - y1, z2 - z1]

        # setup custom properties with tags
        if (obj_props or can_merge) and 'extras' not in node:
            node['extras'] = {}

        for k, v in obj_props.items():
            if node['extras'].get(k):  # tag exists
                tag = node['extras'].get(k)

            if type(v) in (tuple, list, dict):
                tag = json.dumps(v)
            else:
                tag = '{}'.format(v)

            node['extras'][k] = tag

        # if can_merge and 'type' not in obj_props:
        if can_merge:
            node['extras']['type'] = 'Merged'

    def make_empty(self, parent_node, obj):
        gltf_node = {
            'name': obj.name,
        }

        self._setup_node(gltf_node, obj)
        self._add_child(parent_node, gltf_node)

        return gltf_node

    def make_armature(self, parent_node, armature):
        channel = self._buffer.add_channel(
            {
                'componentType': spec.TYPE_FLOAT,
                'type': 'MAT4',
                'extras': {
                    'reference': 'inverseBindMatrices',
                },
            }
        )

        gltf_armature = {
            'name': armature.name,
        }
        gltf_skin = {
            'name': armature.name,
            'joints': [],
            'inverseBindMatrices': channel['bufferView'],
        }
        self._root['skins'].append(gltf_skin)

        set_active_object(armature)

        bone_tails_local = {}
        for bone_name, bone in armature.data.bones.items():
            bone_tails_local[bone_name] = bone.tail

        bone_tails_off = {}
        with Mode('EDIT'):
            for bone_name, bone in armature.data.edit_bones.items():
                bone_tails_off[bone_name] = bone.tail - bone.head

        if self._pose_freeze:
            # set_active_object(armature)

            # disconnect all bones
            with Mode('EDIT'):
                for bone_name, bone in armature.data.edit_bones.items():
                    bone.use_connect = False

            # reset bones rotation
            with Mode('EDIT'):
                for bone_name, bone in armature.data.edit_bones.items():
                    bone.roll = 0
                    bone.length = 10
                    bone.tail = bone.head + mathutils.Vector((0, bone.length, 0))
                    bone.roll = 0

        # create joint nodes
        gltf_joints = {}
        for bone_name, bone in armature.data.bones.items():
            bone_matrix = self._transform(get_bone_matrix(bone, armature))
            bone_tail_matrix = self._transform(mathutils.Matrix.Translation(bone_tails_local[bone_name]))

            if self._pose_freeze:
                bone_matrix = self._freeze(bone_matrix)
                bone_tail_matrix = self._transform(mathutils.Matrix.Translation(bone_tails_off[bone_name]))

            # print(mathutils.Matrix.Translation(bone_tails[bone_name]).to_translation())
            # print((mathutils.Matrix.Translation(bone_tails[bone_name]) @
            #       bone.matrix.to_4x4() @
            #       bone.matrix.to_4x4().inverted()).to_translation())

            gltf_joint = {
                'name': bone_name,
                'rotation': quat_to_list(bone_matrix.to_quaternion()),
                'scale': list(bone_matrix.to_scale()),
                'translation': list(bone_matrix.to_translation()),
                'extras': {
                    'tail': {
                        'translation': list(bone_tail_matrix.to_translation()),
                    }
                }
            }
            gltf_joints[bone_name] = gltf_joint

        # add joints to skin
        for bone_name in sorted(gltf_joints.keys()):
            bone = armature.data.bones[bone_name]
            if bone.parent:  # attach joint to parent joint
                self._add_child(gltf_joints[bone.parent.name], gltf_joints[bone.name])
            else:  # attach joint to armature
                # gltf_skin['skeleton'] = len(self._root['nodes']) - 1
                self._add_child(gltf_armature, gltf_joints[bone.name])

            ib_matrix = self._transform(get_inverse_bind_matrix(bone, armature))
            if self._pose_freeze:
                ib_matrix = self._freeze(ib_matrix)

            self._buffer.write(gltf_skin['inverseBindMatrices'], *matrix_to_list(ib_matrix))
            gltf_skin['joints'].append(len(self._root['nodes']) - 1)

        self._setup_node(gltf_armature, armature)
        self._add_child(parent_node, gltf_armature)

        # no meshes or animation only
        if (not list(filter(is_object_visible, armature.children)) or self._export_type == 'animation'):
            gltf_child_node = {
                'name': '{}_EMPTY'.format(armature.name),
                'skin': len(self._root['skins']) - 1,
            }
            self._add_child(gltf_armature, gltf_child_node)

        return gltf_armature

    def _make_node_mesh(self, parent_node, name, obj=None, can_merge=False):
        """
        Make glTF-node - glTF-mesh pair for chosen Blender object.
        """
        gltf_node = {
            'name': name,
            'extras': {},
        }

        gltf_mesh = None
        need_mesh = False
        if can_merge:
            need_mesh = True
        else:
            if obj.rigid_body is None:
                need_mesh = True
            elif obj.rigid_body.collision_shape == 'MESH':
                need_mesh = True

        if need_mesh:
            gltf_mesh = {
                'name': name,
                'primitives': [],
                'extras': {
                    'targetNames': [],
                },
            }
            self._root['meshes'].append(gltf_mesh)
            gltf_node['mesh'] = len(self._root['meshes']) - 1

        armature = obj and get_armature(obj)
        if armature:
            for i, gltf_skin in enumerate(self._root['skins']):
                if gltf_skin['name'] == armature.name:
                    gltf_node['skin'] = i
                    break

        self._setup_node(gltf_node, obj, can_merge=can_merge)
        self._add_child(parent_node, gltf_node)

        if obj.type == 'MESH':
            for material in obj.data.materials:
                if material and material.node_tree:  # not and empty slot and have nodes tree
                    for node in material.node_tree.nodes:
                        if node.type == 'ATTRIBUTE':
                            gltf_node['extras'][node.attribute_type.lower()] = node.attribute_name

        return gltf_node, gltf_mesh

    def make_mesh(self, parent_node, obj):
        """
        Make mesh-type object.
        """
        gltf_node = None

        # merged nodes
        # if self.can_merge(obj):
        if False:
            collection = get_object_collection(obj)

            for child in self._root['nodes']:
                if child['name'] == collection.name:
                    # got existing glTF node
                    gltf_node = child

                    mesh_id = gltf_node['mesh']
                    # got existing glTF mesh
                    gltf_mesh = self._root['meshes'][mesh_id]
                    break
            else:  # glTF-node - glTF-mesh pair not found
                # create new pair
                gltf_node, gltf_mesh = self._make_node_mesh(parent_node, collection.name, obj, can_merge=True)

            if gltf_mesh:
                self.make_geom(gltf_node, gltf_mesh, obj, can_merge=True)

        # separate nodes
        # if not self.can_merge(obj) or self._keep:
        if True:
            obj_props = get_object_properties(obj)
            if obj_props.get('type') == 'Portal':
                vertices = [list(vertex.co) for vertex in obj.data.vertices]
                gltf_node = {
                    'name': obj.name,
                    'extras': {
                        'vertices': json.dumps(vertices)
                    },
                }
                self._setup_node(gltf_node, obj, can_merge=False)
                self._add_child(parent_node, gltf_node)
                return gltf_node

            gltf_node, gltf_mesh = self._make_node_mesh(parent_node, obj.name, obj, can_merge=False)

            if gltf_mesh:
                self.make_geom(gltf_node, gltf_mesh, obj, can_merge=False)

        return gltf_node

    def make_light(self, parent_node, obj):
        """
        Make light-type object.
        """
        LIGHT_TYPES = {
            'POINT': 'PointLight',
            'SPOT': 'SpotLight',
        }

        gltf_light = {
            'name': obj.name,
            'extras': {
                'type': 'Light',
                'light': LIGHT_TYPES[obj.data.type],
                'color': json.dumps(tuple(obj.data.color)),
                'scale': json.dumps(tuple(obj.scale)),
                'energy': '{:.3f}'.format(obj.data.energy),
                'far': '{:.3f}'.format(obj.data.shadow_soft_size),
            },
        }

        if obj.data.type == 'SPOT':
            gltf_light['extras']['fov'] = '{:.3f}'.format(math.degrees(obj.data.spot_size))

        self._setup_node(gltf_light, obj)
        self._add_child(parent_node, gltf_light)

        return gltf_light

    def convert(self):
        self._buffer = GLTFBuffer(self._output)
        root = super().convert()
        return root, self._buffer

    def write(self, root, output, is_binary=False):
        if is_binary:
            with open(output, 'wb') as f:  # binary mode
                chunk1 = self._buffer.export(root)  # export buffer first because it updates gltf data
                chunk0 = json.dumps(root, indent=4).encode()  # export gltf data
                while len(chunk0) % 4:
                    chunk0 += b' '
                while len(chunk1) % 4:
                    chunk1 += b'\x00'

                # write global headers
                f.write(b'glTF')  # header
                f.write(struct.pack('<I', 2))  # version
                size = (
                    4 + 4 + 4 +  # global headers
                    4 + 4 + len(chunk0) +  # chunk0 + headers
                    4 + 4 + len(chunk1)
                )  # chunk1 + headers
                f.write(struct.pack('<I', size))  # full size

                # write chunk0 with headers
                f.write(struct.pack('<I', len(chunk0)))
                f.write(b'JSON')
                f.write(chunk0)

                # write chunk1 with headers
                f.write(struct.pack('<I', len(chunk1)))
                f.write(b'BIN\0')
                f.write(chunk1)

        else:
            with open(output, 'w') as f:  # text mode
                json.dump(root, f, indent=4)


class GLTFExporterOperator(bpy.types.Operator, ExportHelper):
    bl_idname = 'scene.gltf'
    bl_label = 'Export glTF'
    bl_description = 'Export glTF'
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = '.gltf'
    filter_glob: bpy.props.StringProperty(default='*.gltf', options={'HIDDEN'})

    def execute(self, context: bpy.types.Context):
        if not self.filepath:
            return {'CANCELLED'}

        class Args(object):
            inputs = []
            output = self.filepath
            export = 'all'
            render = 'default'
            exec = None
            action = None
            speed = None
            scale = None
            merge = None
            keep = None
            no_extra_uv = None
            no_materials = None
            no_textures = None
            empty_textures = None
            set_origin = None
            normalize_weights = None

        args = Args()
        e = GLTFExporter(args)
        out, buf = e.convert()

        if args.output.endswith('.gltf'):
            # write buffer into separate file
            buf.export(out, args.output.replace('.gltf', '.bin'))
            # write glTF data
            e.write(out, args.output, is_binary=False)

        else:
            # write glTF data with embedded buffer
            e.write(out, args.output, is_binary=True)

        # re-open current file
        bpy.ops.wm.open_mainfile(filepath=bpy.data.filepath)

        return {"FINISHED"}

    def invoke(self, context, event):
        return cast(Set[str], ExportHelper.invoke(self, context, event))

    def draw(self, context):
        pass


def export(export_op, context):
    export_op.layout.operator(GLTFExporterOperator.bl_idname, text='glTF using KITSUNETSUKI Asset Tools (.gltf)')
