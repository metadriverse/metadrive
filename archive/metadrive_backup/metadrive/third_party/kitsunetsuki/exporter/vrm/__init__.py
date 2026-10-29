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
import configparser
import os

from bpy_extras.io_utils import ExportHelper
from typing import Set, cast

from ..gltf import GLTFExporter
from ..gltf import spec
from .armature import ArmatureMixin

BLENDSHAPE_PRESETS = (
    'neutral',
    'a',
    'i',
    'u',
    'e',
    'o',
    'blink',
    'joy',
    'angry',
    'sorrow',
    'fun',
    'lookup',
    'lookdown',
    'lookleft',
    'lookright',
    'blink_l',
    'blink_r',
)


class VRMExporter(ArmatureMixin, GLTFExporter):
    def __init__(self, args):
        super().__init__(args)

        self._z_up = False
        self._pose_freeze = True
        self._export_type = 'all'

    def _add_vrm_thumbnail(self, gltf_node, filepath):
        gltf_sampler = {
            'name': os.path.basename(filepath),
            'wrapS': spec.CLAMP_TO_EDGE,
            'wrapT': spec.CLAMP_TO_EDGE,
        }
        gltf_node['samplers'].append(gltf_sampler)

        gltf_image = {
            'name': os.path.basename(filepath),
            'mimeType': 'image/png',
            'extras': {
                'uri': filepath,
            }
        }
        gltf_node['images'].append(gltf_image)

        gltf_texture = {
            'sampler': len(gltf_node['samplers']) - 1,
            'source': len(gltf_node['images']) - 1,
        }
        gltf_node['textures'].append(gltf_texture)

        texid = len(gltf_node['textures']) - 1
        gltf_node['extensions']['VRM']['meta']['texture'] = texid

    def make_root_node(self):
        gltf_node = super().make_root_node()

        data = {}
        text = bpy.data.texts.get('VRM.ini')
        if text:
            data = configparser.ConfigParser()
            data.read_string(text.as_string())
        else:
            raise RuntimeError('Missing "VRM.ini" text block.')

        vrm_meta = {
            'exporterVersion': gltf_node['asset']['generator'],
            'specVersion': '0.0',
            'meta': {
                'title': data['meta']['title'],
                'version': data['meta']['version'],
                'author': data['meta']['author'],
                'contactInformation': data['meta']['contactInformation'],
                'reference': data['meta']['reference'],
                'texture': 0,  # thumbnail texture
                'allowedUserName': data['meta'].get('allowedUserName', 'OnlyAuthor'),
                'violentUssageName': data['meta'].get('violentUssageName', 'Disallow'),
                'sexualUssageName': data['meta'].get('sexualUssageName', 'Disallow'),
                'commercialUssageName': data['meta'].get('commercialUssageName', 'Disallow'),
                'otherPermissionUrl': data['meta'].get('otherPermissionUrl', ''),
                'licenseName': data['meta'].get('licenseName', 'Redistribution_Prohibited'),
                'otherLicenseUrl': data['meta'].get('otherLicenseUrl', ''),
            },
            'humanoid': {
                'armStretch': 0.0,
                'legStretch': 0.0,
                'lowerArmTwist': 0.0,  # LowerArm bone roll
                'upperArmTwist': 0.0,  # UpperArm bone roll
                'lowerLegTwist': 0.0,  # LowerLeg bone roll
                'upperLegTwist': 0.0,  # UpperLeg bone roll
                'feetSpacing': 0.0,
                'hasTranslationDoF': False,
                'humanBones': [],
            },
            'firstPerson': {
                'firstPersonBone': None,
                'firstPersonBoneOffset': {
                    'x': 0,
                    'y': 0,
                    'z': 0,
                },
                'meshAnnotations': [],
                'lookAtTypeName': 'Bone',
                # 'lookAtTypeName': 'BlendShape',
                'lookAtHorizontalInner': None,
                'lookAtHorizontalOuter': None,
                'lookAtVerticalDown': None,
                'lookAtVerticalUp': None,
            },
            'blendShapeMaster': {
                'blendShapeGroups': [],
            },
            'secondaryAnimation': {
                'boneGroups': [],
                # 'colliderGroups': [],
            },
            'materialProperties': [],
        }

        gltf_node['extensionsUsed'].append('VRM')
        gltf_node['extensions']['VRM'] = vrm_meta
        gltf_node['materials'] = []

        # make thumbnail
        if self._inputs:
            prefix = os.path.basename(self._inputs[0]).replace('.blend', '.png')
            inpdir = os.path.dirname(os.path.abspath(self._inputs[0]))
            if os.path.exists(inpdir) and os.path.isdir(inpdir):
                for filename in reversed(sorted(os.listdir(inpdir))):
                    if filename.startswith(prefix):
                        self._add_vrm_thumbnail(gltf_node, os.path.join(inpdir, filename))
                        break

        return gltf_node

    def _make_vrm_material(self, material):
        vrm_material = {
            'floatProperties': {
                '_BlendMode': 0 if material.blend_method == 'OPAQUE' else 1,
                '_BumpScale': 1,
                '_CullMode': 2 if material.use_backface_culling else 0,
                '_Cutoff': material.alpha_threshold,
                '_DebugMode': 0,
                '_DstBlend': 0,
                '_IndirectLightIntensity': 0.1,
                '_LightColorAttenuation': 0,
                '_MToonVersion': 35,
                '_OutlineColorMode': 0,
                '_OutlineCullMode': 1,
                '_OutlineLightingMix': 1,
                '_OutlineScaledMaxDistance': 1,
                '_OutlineWidth': 0.5,
                '_OutlineWidthMode': 0,
                '_ReceiveShadowRate': 1,
                '_RimFresnelPower': 1,
                '_RimLift': 0,
                '_RimLightingMix': 0,
                '_ShadeShift': 0,
                '_ShadeToony': 0.9,
                '_ShadingGradeRate': 1,
                '_SrcBlend': 1,
                '_UvAnimRotation': 0,
                '_UvAnimScrollX': 0,
                '_UvAnimScrollY': 0,
                '_ZWrite': 1,
            },
            'keywordMap': {},
            'name': material.name,
            'renderQueue': 2000,
            'shader': 'VRM_USE_GLTFSHADER',
            'tagMap': {},
            'textureProperties': {},
            'vectorProperties': {
                '_BumpMap': [0, 0, 1, 1],
                '_Color': [1, 1, 1, 1],
                '_EmissionColor': [0, 0, 0, 1],
                '_EmissionMap': [0, 0, 1, 1],
                '_MainTex': [0, 0, 1, 1],
                '_OutlineColor': [0, 0, 0, 1],
                '_OutlineWidthTexture': [0, 0, 1, 1],
                '_ReceiveShadowTexture': [0, 0, 1, 1],
                '_RimColor': [0, 0, 0, 1],
                '_RimTexture': [0, 0, 1, 1],
                '_ShadeColor': [1, 1, 1, 1],
                '_ShadeTexture': [0, 0, 1, 1],
                '_ShadingGradeTexture': [0, 0, 1, 1],
                '_SphereAdd': [0, 0, 1, 1],
                '_UvAnimMaskTexture': [0, 0, 1, 1],
            },
        }

        return vrm_material

    def _make_vrm_blend_shape(self, name):
        """
        Standby expression:
        - Neutral

        Lip-sync:
        - A (aa)
        - I (ih)
        - U (ou)
        - E (e)
        - O (oh)

        Blink:
        - Blink
        - Blink_L
        - Blink_R

        Emotion:
        - Fun
        - Angry
        - Sorrow
        - Joy

        Eye control:
        - LookUp
        - LookDown
        - LookLeft
        - LookRight
        """

        vrm_name = {
            # try to get VRM blend shapes from VRChat
            'vrc.v_aa': 'A',
            'vrc.v_ih': 'I',
            'vrc.v_ou': 'U',
            'vrc.v_e': 'E',
            'vrc.v_oh': 'O',
            'vrc.blink': 'Blink',
            'vrc.blink_left': 'Blink_L',
            'vrc.blink_right': 'Blink_R',
        }.get(name, name)

        preset = 'unknown'
        if vrm_name.lower() in BLENDSHAPE_PRESETS:
            preset = vrm_name.lower()

        vrm_blend_shape = {
            'name': vrm_name,
            'presetName': preset,
            'isBinary': False,
            'binds': [],  # bind to mesh ID and shape key ID with shape key weight
            'materialValues': [],  # material values override
        }

        return vrm_blend_shape

    def convert(self):
        root, buffer_ = super().convert()

        for gltf_material_id, gltf_material in enumerate(root['materials']):
            material = bpy.data.materials[gltf_material['name']]
            vrm_material = self._make_vrm_material(material)

            if gltf_material['alphaMode'] == 'OPAQUE':
                vrm_material['tagMap']['RenderType'] = 'Opaque'
                vrm_material['shader'] = 'VRM/MToon'
            else:
                vrm_material['shader'] = 'VRM/UnlitCutout'

            if gltf_material['pbrMetallicRoughness'].get('baseColorTexture'):
                vrm_material['textureProperties']['_MainTex'] = gltf_material['pbrMetallicRoughness']['baseColorTexture'
                                                                                                      ]['index']

            root['extensions']['VRM']['materialProperties'].append(vrm_material)

        vrm_blend_shapes = {
            'Neutral': {
                'name': 'Neutral',
                'presetName': 'neutral',
                'isBinary': False,
                'binds': [],
                'materialValues': [],
            }
        }
        for gltf_mesh_id, gltf_mesh in enumerate(root['meshes']):
            vrm_annotation = {
                'firstPersonFlag': 'Auto',
                'mesh': gltf_mesh_id,
            }
            root['extensions']['VRM']['firstPerson']['meshAnnotations'].append(vrm_annotation)

            for gltf_primitive_id, gltf_primitive in enumerate(gltf_mesh['primitives']):
                for sk_id, sk_name in enumerate(gltf_primitive['extras']['targetNames']):
                    if sk_name in vrm_blend_shapes:
                        vrm_blend_shape = vrm_blend_shapes[sk_name]
                    else:
                        vrm_blend_shape = self._make_vrm_blend_shape(sk_name)
                        vrm_blend_shapes[sk_name] = vrm_blend_shape

                    for vrm_bind in vrm_blend_shape['binds']:
                        if vrm_bind['mesh'] == gltf_mesh_id and vrm_bind['index'] == sk_id:
                            break
                    else:
                        vrm_bind = {
                            'mesh': gltf_mesh_id,
                            'index': sk_id,
                            'weight': 100,
                        }
                        vrm_blend_shape['binds'].append(vrm_bind)

        for vrm_blend_shape in vrm_blend_shapes.values():
            root['extensions']['VRM']['blendShapeMaster']['blendShapeGroups'].append(vrm_blend_shape)

        return root, buffer_


class VRMExporterOperator(bpy.types.Operator, ExportHelper):
    bl_idname = 'avatar.vrm'
    bl_label = 'Export VRM'
    bl_description = 'Export VRM'
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = '.vrm'
    filter_glob: bpy.props.StringProperty(default='*.vrm', options={'HIDDEN'})

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
        e = VRMExporter(args)
        out, buf = e.convert()

        e.write(out, args.output, is_binary=True)

        # re-open current file
        bpy.ops.wm.open_mainfile(filepath=bpy.data.filepath)

        return {"FINISHED"}

    def invoke(self, context, event):
        return cast(Set[str], ExportHelper.invoke(self, context, event))

    def draw(self, context):
        pass


def export(export_op, context):
    export_op.layout.operator(VRMExporterOperator.bl_idname, text='VRM using KITSUNETSUKI Asset Tools (.vrm)')
