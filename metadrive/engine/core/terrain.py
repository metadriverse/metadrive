# import numpyf
import math

import os
import pathlib
import sys

#
#
from abc import ABC
from metadrive.constants import TerrainProperty, CameraTagStateKey
import cv2
import numpy as np
from panda3d.bullet import BulletRigidBodyNode, BulletPlaneShape
from panda3d.bullet import ZUp, BulletHeightfieldShape
from panda3d.core import SamplerState, PNMImage, CardMaker, LQuaternionf, NodePath
from panda3d.core import Vec3, ShaderTerrainMesh, Texture, TextureStage, Shader, Filename

from metadrive.base_class.base_object import BaseObject
from metadrive.constants import CamMask, Semantics
from metadrive.constants import MetaDriveType, CollisionGroup
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.logger import get_logger
from metadrive.third_party.diamond_square import diamond_square
from metadrive.utils.utils import is_win

logger = get_logger()


class Terrain(BaseObject, ABC):
    """
    Terrain and map
    """
    COLLISION_MASK = CollisionGroup.Terrain
    HEIGHT = 0.0
    PROBE_HEIGHT = 600
    PROBE_SIZE = 1024
    SEMANTIC_LABEL = Semantics.TERRAIN.label
    PATH = pathlib.PurePosixPath(__file__).parent if not is_win() else pathlib.Path(__file__).resolve().parent

    def __init__(self, show_terrain, engine):
        super(Terrain, self).__init__(random_seed=0)
        self.origin.hide(CamMask.MiniMap | CamMask.Shadow)
        # use plane terrain or mesh terrain， True by default.
        self.use_mesh_terrain = engine.global_config["use_mesh_terrain"]
        self.full_size_mesh = engine.global_config["full_size_mesh"]

        # collision mesh
        self.plane_collision_terrain = None  # a flat collision shape
        self.mesh_collision_terrain = None  # a 3d mesh, Not available yet!

        # visualization mesh feature
        self._terrain_size = TerrainProperty.terrain_size  # [m]
        self._height_scale = engine.global_config["height_scale"]  # [m]
        self._drivable_area_extension = engine.global_config["drivable_area_extension"]  # [m] road marin
        # it should include the whole map. Otherwise, road will have no texture!
        self._heightmap_size = self._semantic_map_size = TerrainProperty.map_region_size  # [m]
        self._heightfield_start = int((self._terrain_size - self._heightmap_size) / 2)
        self._semantic_map_pixel_per_meter = TerrainProperty.get_semantic_map_pixel_per_meter()  # [m]  pixels per meter
        self._terrain_offset = 2055  # 1023/65536 * self._height_scale [m] warning: make it power 2 -1!
        # pre calculate some variables
        self._elevation_texture_ratio = self._terrain_size / self._semantic_map_size  # for shader
        self.origin.setZ(-(self._terrain_offset + 1) / 65536 * self._height_scale * 2)

        self._mesh_terrain = None
        self._mesh_terrain_height = None
        self._mesh_terrain_node = None
        self._terrain_shader_set = False  # only set once
        self.probe = None

        self.render = self.render and show_terrain

        if self.render:
            # if engine.use_render_pipeline:
            self._load_mesh_terrain_textures(engine)
            self._mesh_terrain_node = ShaderTerrainMesh()
            # disable env probe as some vehicle models may break it
            # self.probe = engine.render_pipeline.add_environment_probe()
            # self.probe.set_pos(0, 0, self.PROBE_HEIGHT)
            # self.probe.set_scale(self.PROBE_SIZE * 2, self.PROBE_SIZE * 2, 1000)

        if self.use_mesh_terrain or self.render:
            self._load_height_field_image(engine)

    def before_reset(self):
        """
        Clear existing terrain
        Returns: None

        """
        # detach current map
        assert self.engine is not None, "Can not call this without initializing engine"
        if self.is_attached():
            self.detach_from_world(self.engine.physics_world)

    # @time_me
    def reset(self, center_point):
        """
        Update terrain according to current map
        """
        if not self.use_mesh_terrain and self.plane_collision_terrain is None:
            # only generate once if plane terrain
            self.generate_plane_collision_terrain()

        if self.render or self.use_mesh_terrain:
            # modify default height image
            drivable_region = self.get_drivable_region(center_point)

            # embed to the original height image
            start = self._heightfield_start
            end = self._heightfield_start + self._heightmap_size
            heightfield_base = np.copy(self.heightfield_img)

            if abs(np.mean(drivable_region) - 0.0) < 1e-3:
                heightfield_to_modify = heightfield_base[start:end, start:end, ...]
                logger.warning(
                    "No map is found in map region, "
                    "size: [{}, {}], "
                    "center: {}".format(self._semantic_map_size, self._semantic_map_size, center_point)
                )
            else:
                drivable_area_height_mean = np.mean(
                    self.heightfield_img[start:end, start:end, ...][np.where(drivable_region)]
                )
                heightfield_base = np.where(
                    heightfield_base > (drivable_area_height_mean - self._terrain_offset),
                    heightfield_base - (drivable_area_height_mean - self._terrain_offset), 0
                ).astype(np.uint16)
                heightfield_to_modify = heightfield_base[start:end, start:end, ...]
                heightfield_base[start:end, start:end,
                                 ...] = np.where(drivable_region, self._terrain_offset, heightfield_to_modify)

            # generate collision mesh
            if self.use_mesh_terrain:
                self._generate_collision_mesh(
                    heightfield_base if self.full_size_mesh else heightfield_to_modify, self._height_scale
                )

            if self.render:
                # Make semantics for shader terrain
                semantics = self.get_terrain_semantics(center_point)
                semantic_tex = Texture()
                semantic_tex.setup2dTexture(*semantics.shape[:2], Texture.TFloat, Texture.F_red)
                semantic_tex.setRamImage(semantics)

                # to panda texture
                heightfield_tex = Texture()
                heightfield_tex.setup2dTexture(*heightfield_base.shape[:2], Texture.TShort, Texture.FLuminance)
                heightfield_tex.setRamImage(heightfield_base)

                # generate terrain visualization
                self._generate_mesh_vis_terrain(self._terrain_size, heightfield_tex, semantic_tex)
        # reset position
        self.set_position(center_point)
        self.attach_to_world(self.engine.render, self.engine.physics_world)

    def generate_plane_collision_terrain(self):
        """
        generate a plane as terrain
        Returns:

        """
        # Create once for lazy-reset
        # If no render pipeline, we can only have 2d terrain. It will only be generated for once.
        shape = BulletPlaneShape(Vec3(0, 0, 1), 0)
        node = BulletRigidBodyNode(MetaDriveType.GROUND)
        node.setFriction(.9)
        node.addShape(shape)

        node.setIntoCollideMask(self.COLLISION_MASK)
        self.dynamic_nodes.append(node)

        self.plane_collision_terrain = self.origin.attachNewNode(node)
        self.plane_collision_terrain.setZ(-self.origin.getZ())
        self._node_path_list.append(np)

    def _generate_mesh_vis_terrain(
        self,
        size,
        heightfield: Texture,
        attribute_tex: Texture,
        target_triangle_width=10,
        engine=None,
    ):
        """
        Given a height field map to generate terrain and an attribute_tex to texture terrain, so we can get road/grass
        pixels_per_meter is determined by heightfield.size/size
        lane line and so on.
        :param size: [m] this terrain of the generated terrain
        :param heightfield: terrain heightfield. It should be a 16-bit png and have a quadratic size of a power of two.
        :param attribute_tex: doing texture splatting. r,g,b,a represent: grass/road/lane_line ratio respectively
        :param target_triangle_width: For a value of 10.0 for example, the terrain will attempt to make every triangle 10 pixels wide on screen.
        :param engine: engine instance
        :return:
        """
        engine = engine or self.engine
        assert engine is not None
        assert isinstance(heightfield, Texture), "heightfield file must be Panda3D Texture"

        # Set a heightfield,
        heightfield.wrap_u = SamplerState.WM_clamp
        heightfield.wrap_v = SamplerState.WM_clamp
        self._mesh_terrain_node.heightfield = heightfield

        # Set the target triangle width.
        self._mesh_terrain_node.target_triangle_width = target_triangle_width
        self._mesh_terrain_node.setTargetTriangleWidth(target_triangle_width)
        # self._mesh_terrain_node.setUpdateEnabled(False)
        self._mesh_terrain_node.setChunkSize(128)
        # self._mesh_terrain_node.setChunkSize(64)

        # Generate the terrain
        self._mesh_terrain_node.generate()
        self._mesh_terrain = self.origin.attach_new_node(self._mesh_terrain_node)
        # shader is determined by tag state, enabling multi-pass rendering
        self._mesh_terrain.setTag(CameraTagStateKey.Semantic, self.SEMANTIC_LABEL)
        self._mesh_terrain.setTag(CameraTagStateKey.RGB, self.SEMANTIC_LABEL)
        self._mesh_terrain.setTag(CameraTagStateKey.Depth, self.SEMANTIC_LABEL)
        self._set_terrain_shader(engine, attribute_tex)

        # Attach the terrain to the main scene and set its scale. With no scale
        # set, the terrain ranges from (0, 0, 0) to (1, 1, 1)
        self._mesh_terrain.set_scale(size, size, self._height_scale)
        self._mesh_terrain.set_pos(-size / 2, -size / 2, 0)

    def _set_terrain_shader(self, engine, attribute_tex):
        """
        Set a shader on the terrain. The ShaderTerrainMesh only works with an applied shader. You can use the shaders
        used here in your own application.

        Note: you have to make sure you modified the terrain_effect.yaml and vert.glsl/frag.glsl together, as they are
        made for different render pipeline. We expect the same terrain generated from different pipelines.
        """
        if not self._terrain_shader_set:
            if engine.use_render_pipeline:
                engine.render_pipeline.reload_shaders()
                terrain_effect = AssetLoader.file_path("../shaders", "terrain_effect.yaml")
                engine.render_pipeline.set_effect(self._mesh_terrain, terrain_effect, {}, 100)

            # # height
            self._mesh_terrain.set_shader_input("camera", self.engine.camera)
            self._mesh_terrain.set_shader_input("height_scale", self._height_scale)

            # grass
            self._mesh_terrain.set_shader_input("grass_tex", self.grass_tex)
            self._mesh_terrain.set_shader_input("grass_normal", self.grass_normal)
            self._mesh_terrain.set_shader_input("grass_rough", self.grass_rough)
            self._mesh_terrain.set_shader_input("grass_tex_ratio", self.grass_tex_ratio)
            #
            # # side
            # self._mesh_terrain.set_shader_input("side_tex", self.side_tex)
            # self._mesh_terrain.set_shader_input("side_normal", self.side_normal)

            # road
            self._mesh_terrain.set_shader_input("rock_tex", self.rock_tex)
            self._mesh_terrain.set_shader_input("rock_normal", self.rock_normal)
            self._mesh_terrain.set_shader_input("rock_rough", self.rock_rough)
            self._mesh_terrain.set_shader_input("rock_tex_ratio", self.rock_tex_ratio)

            self._mesh_terrain.set_shader_input("road_tex", self.road_texture)
            self._mesh_terrain.set_shader_input("yellow_tex", self.yellow_lane_line)
            self._mesh_terrain.set_shader_input("white_tex", self.white_lane_line)
            self._mesh_terrain.set_shader_input("road_normal", self.road_texture_normal)
            self._mesh_terrain.set_shader_input("road_rough", self.road_texture_rough)
            self._mesh_terrain.set_shader_input("elevation_texture_ratio", self._elevation_texture_ratio)

            # crosswalk
            self._mesh_terrain.set_shader_input("crosswalk_tex", self.crosswalk_tex)
            self._terrain_shader_set = True

            # semantic color input
            def to_float(color):
                return Vec3(*[i / 255 for i in color])

            self._mesh_terrain.set_shader_inputs(
                crosswalk_semantics=to_float(Semantics.CROSSWALK.color),
                lane_line_semantics=to_float(Semantics.LANE_LINE.color),
                road_semantics=to_float(Semantics.ROAD.color),
                ground_semantics=to_float(Semantics.TERRAIN.color)
            )
        self._mesh_terrain.set_shader_input("attribute_tex", attribute_tex)

    def reload_terrain_shader(self):
        """
        Reload terrain shader for debug/develop
        Returns: None

        """
        vert = AssetLoader.file_path("../shaders", "terrain.vert.glsl")
        frag = AssetLoader.file_path("../shaders", "terrain.frag.glsl")
        terrain_shader = Shader.load(Shader.SL_GLSL, vert, frag)
        self._mesh_terrain.clear_shader()
        self._mesh_terrain.set_shader(terrain_shader)

    def _generate_collision_mesh(self, heightfield_img, height_scale):
        """
        Work in Progress
        Args:
            heightfield_img:
            height_scale:

        Returns:

        """
        # clear previous mesh
        self.dynamic_nodes.clear()
        mesh = heightfield_img
        mesh = np.flipud(mesh)
        mesh = cv2.resize(mesh, (mesh.shape[0] + 1, mesh.shape[1] + 1))
        path_to_store = self.PATH.joinpath("run_time_map_mesh_{}.png".format(self.engine.pid))
        cv2.imencode('.png', mesh)[1].tofile(path_to_store)
        # cv2.imwrite(str(path_to_store), mesh)
        p = PNMImage(Filename(str(AssetLoader.windows_style2unix_style(path_to_store) if is_win() else path_to_store)))
        os.remove(path_to_store)  # remove after using

        shape = BulletHeightfieldShape(p, height_scale * 2, ZUp)
        shape.setUseDiamondSubdivision(True)

        node = BulletRigidBodyNode(MetaDriveType.GROUND)
        node.setFriction(.9)
        node.addShape(shape)

        node.setIntoCollideMask(CollisionGroup.Terrain)
        self.dynamic_nodes.append(node)

        self.mesh_collision_terrain = self.origin.attachNewNode(node)

    def set_position(self, position, height=None):
        """
        Set the terrain position to the map center
        Args:
            position: position in world coordinates
            height: Placeholder, No effect

        Returns:

        """
        if self.render:
            pos = (self._mesh_terrain.get_pos()[0] + position[0], self._mesh_terrain.get_pos()[1] + position[1])
            self._mesh_terrain.set_pos(*pos, 0)
            if self.probe is not None:
                self.probe.set_pos(*pos, self.PROBE_HEIGHT)
        if self.use_mesh_terrain:
            self.mesh_collision_terrain.set_pos(*position[:2], self._height_scale)

    def _generate_card_terrain(self):
        """
        Generate a 2D-card terrain, which is deprecated
        Returns:

        """
        raise DeprecationWarning
        self.origin.hide(
            CamMask.MiniMap | CamMask.Shadow | CamMask.DepthCam | CamMask.ScreenshotCam | CamMask.SemanticCam
        )
        # self.terrain_normal = self.loader.loadTexture(
        #     AssetLoader.file_path( "textures", "grass2", "normal.jpg")
        # )
        self.terrain_texture = self.loader.loadTexture(AssetLoader.file_path("textures", "ground.png"))
        self.terrain_texture.set_format(Texture.F_srgb)
        self.terrain_texture.setWrapU(Texture.WM_repeat)
        self.terrain_texture.setWrapV(Texture.WM_repeat)
        self.ts_color = TextureStage("color")
        self.ts_normal = TextureStage("normal")
        self.ts_normal.set_mode(TextureStage.M_normal)
        # self.set_position((0, 0), self.HEIGHT)
        cm = CardMaker('card')
        scale = 4000
        cm.setUvRange((0, 0), (scale / 10, scale / 10))
        card = self.origin.attachNewNode(cm.generate())

        self._node_path_list.append(card)

        card.set_scale(scale)
        card.setPos(-scale / 2, -scale / 2, -0.1)
        card.setZ(-.05)
        card.setTexture(self.ts_color, self.terrain_texture)
        # card.setTexture(self.ts_normal, self.terrain_normal)
        self.terrain_texture.setMinfilter(SamplerState.FT_linear_mipmap_linear)
        self.terrain_texture.setAnisotropicDegree(8)
        card.setQuat(LQuaternionf(math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0, 0))

    def _load_height_field_image(self, engine):
        # basic height_field
        heightfield_tex = engine.loader.loadTexture(AssetLoader.file_path("textures", "terrain", "heightfield.png"))
        heightfield_img = np.frombuffer(heightfield_tex.getRamImage().getData(), dtype=np.uint16)
        heightfield_img = heightfield_img.reshape((heightfield_tex.getYSize(), heightfield_tex.getXSize(), 1))
        down_sample_rate = int(heightfield_tex.getYSize() / self._terrain_size)  # downsample to 2048 m
        self.heightfield_img = np.array(heightfield_img[::down_sample_rate, ::down_sample_rate])

    def _load_mesh_terrain_textures(self, engine, anisotropic_degree=16, filter_type=Texture.FTLinearMipmapLinear):
        """
        Only maintain one copy of all asset
        :param anisotropic_degree: https://docs.panda3d.org/1.10/python/programming/texturing/texture-filter-types
        :param filter_type: https://docs.panda3d.org/1.10/python/programming/texturing/texture-filter-types
        :return: None
        """
        # texture stage
        self.ts_color = TextureStage("color")
        self.ts_normal = TextureStage("normal")
        self.ts_normal.setMode(TextureStage.M_normal)

        # grass
        # if engine.use_render_pipeline:
        #     # grass
        #     self.grass_tex = self.loader.loadTexture(
        #         AssetLoader.file_path("textures", "grass2", "grass_path_2_diff_1k.png")
        #     )
        #     self.grass_normal = self.loader.loadTexture(
        #         AssetLoader.file_path("textures", "grass2", "grass_path_2_nor_gl_1k.png")
        #     )
        #     self.grass_rough = self.loader.loadTexture(
        #         AssetLoader.file_path("textures", "grass2", "grass_path_2_rough_1k.png")
        #     )
        #     self.grass_tex_ratio = 128.0
        # else:
        self.grass_tex = self.loader.loadTexture(
            AssetLoader.file_path("textures", "grass1", "GroundGrassGreen002_COL_1K.jpg")
        )
        self.grass_normal = self.loader.loadTexture(
            AssetLoader.file_path("textures", "grass1", "GroundGrassGreen002_NRM_1K.jpg")
        )
        self.grass_rough = self.loader.loadTexture(
            AssetLoader.file_path("textures", "grass2", "grass_path_2_rough_1k.png")
        )
        self.grass_tex_ratio = 64

        v_wrap = Texture.WMRepeat
        u_warp = Texture.WMMirror

        for tex in [self.grass_tex, self.grass_normal, self.grass_rough]:
            tex.set_wrap_u(u_warp)
            tex.set_wrap_v(v_wrap)
            tex.setMinfilter(filter_type)
            tex.setMagfilter(filter_type)
            tex.setAnisotropicDegree(anisotropic_degree)

        # rock
        self.rock_tex = self.loader.loadTexture(
            AssetLoader.file_path("textures", "rock", "brown_mud_leaves_01_diff_1k.png")
        )
        self.rock_normal = self.loader.loadTexture(
            AssetLoader.file_path("textures", "rock", "brown_mud_leaves_01_nor_gl_1k.png")
        )
        self.rock_rough = self.loader.loadTexture(
            AssetLoader.file_path("textures", "rock", "brown_mud_leaves_01_rough_1k.png")
        )
        self.rock_tex_ratio = 128

        v_wrap = Texture.WMRepeat
        u_warp = Texture.WMMirror

        for tex in [self.rock_tex, self.rock_normal, self.rock_rough]:
            tex.set_wrap_u(u_warp)
            tex.set_wrap_v(v_wrap)
            tex.setMinfilter(filter_type)
            tex.setMagfilter(filter_type)
            tex.setAnisotropicDegree(anisotropic_degree)

        # # sidewalk
        # self.side_tex = self.loader.loadTexture(AssetLoader.file_path("textures", "sidewalk", "color.png"))
        # self.side_normal = self.loader.loadTexture(AssetLoader.file_path("textures", "sidewalk", "normal.png"))
        #
        # v_wrap = Texture.WMRepeat
        # u_warp = Texture.WMMirror
        #
        # for tex in [self.side_tex, self.side_normal]:
        #     tex.set_wrap_u(u_warp)
        #     tex.set_wrap_v(v_wrap)
        #     tex.setMinfilter(filter_type)
        #     tex.setMagfilter(filter_type)
        #     tex.setAnisotropicDegree(anisotropic_degree)

        # Road surface
        # self.road_texture = self.loader.loadTexture(AssetLoader.file_path("textures", "sci", "new_color.png"))
        self.road_texture = self.loader.loadTexture(AssetLoader.file_path("textures", "asphalt", "diff_2k.png"))
        self.road_texture_normal = self.loader.loadTexture(
            AssetLoader.file_path("textures", "asphalt", "normal_2k.png")
        )
        self.road_texture_rough = self.loader.loadTexture(AssetLoader.file_path("textures", "asphalt", "rough_2k.png"))
        v_wrap = Texture.WMRepeat
        u_warp = Texture.WMMirror
        filter_type = Texture.FTLinearMipmapLinear
        anisotropic_degree = 16
        for tex in [self.road_texture_rough, self.road_texture, self.road_texture_normal]:
            tex.set_wrap_u(u_warp)
            tex.set_wrap_v(v_wrap)
            tex.setMinfilter(filter_type)
            tex.setMagfilter(filter_type)
            tex.setAnisotropicDegree(anisotropic_degree)
        # self.road_texture_displacement = self.loader.loadTexture(AssetLoader.file_path("textures", "sci", "normal.jpg"))
        # self.road_texture.setMinfilter(minfilter)
        # self.road_texture.setAnisotropicDegree(anisotropic_degree)

        # lane line
        white_lane_line = PNMImage(1024, 1024, 4)
        white_lane_line.fill(1., 1., 1.)
        self.white_lane_line = Texture("white lane line")
        self.white_lane_line.load(white_lane_line)

        yellow_lane_line = PNMImage(1024, 1024, 4)
        yellow_lane_line.fill(*(255 / 255, 200 / 255, 0 / 255))
        self.yellow_lane_line = Texture("white lane line")
        self.yellow_lane_line.load(yellow_lane_line)

        # crosswalk
        tex = np.frombuffer(self.road_texture.getRamImage().getData(), dtype=np.uint8)
        tex = tex.copy()
        tex = tex.reshape((self.road_texture.getYSize(), self.road_texture.getXSize(), 3))
        step_size = 64
        for x in range(0, 2048, step_size * 2):
            tex[x:x + step_size, ...] = 220
        self.crosswalk_tex = Texture()
        self.crosswalk_tex.setup2dTexture(*tex.shape[:2], Texture.TUnsignedByte, Texture.F_rgb)
        self.crosswalk_tex.setRamImage(tex)
        # self.crosswalk_tex.write("test_crosswalk.png")

    def _make_random_terrain(self, texture_size, terrain_size, heightfield):
        """
        Deprecated
        Args:
            texture_size:
            terrain_size:
            heightfield:

        Returns:

        """
        raise DeprecationWarning
        height_1 = width_2 = height_3 = width_3 = length = int((terrain_size - texture_size) / 2)
        width_1 = height_2 = width = texture_size
        max_height = 8192
        min_height = 0
        roughness = 0.14

        array_1 = diamond_square(
            [height_1, width_1], min_height, max_height, roughness, random_seed=self.generate_seed()
        )
        array_2 = diamond_square(
            [height_2, width_2], min_height, max_height, roughness, random_seed=self.generate_seed()
        )

        array_3 = diamond_square(
            [height_3, width_3], min_height, max_height, roughness, random_seed=self.generate_seed()
        )

        heightfield[:length, length:length + width] = array_1
        heightfield[-length:, length:length + width] = array_1

        heightfield[length:length + width, :length] = array_2
        heightfield[length:length + width, -length:] = array_2

        heightfield[:length, :length] = array_3
        heightfield[-length:, :length] = array_3
        heightfield[:length, -length:] = array_3
        heightfield[-length:, -length:] = array_3

    @property
    def mesh_terrain(self):
        """
        Exposing the mesh_terrain for outside use, i.e. shadow caster
        Returns: mesh_terrain node path

        """
        return self._mesh_terrain

    def get_drivable_region(self, center_point):
        """
        Get drivable area, consisting of all roads in map
        Returns: drivable area

        """
        if self.engine.current_map:
            drivable_region = self.engine.current_map.get_height_map(
                center_point, self._heightmap_size, 1, self._drivable_area_extension
            )
        else:
            drivable_region = np.ones((self._heightmap_size, self._heightmap_size, 1))
        return drivable_region

    def get_terrain_semantics(self, center_point):
        """
        Return semantic maps indicating the property of the terrain for specific region
        Returns:

        """
        layer = ["lane", "lane_line"]
        if self.engine.global_config["show_crosswalk"]:
            layer.append("crosswalk")
        if self.engine.current_map:
            semantics = self.engine.current_map.get_semantic_map(
                center_point,
                size=self._semantic_map_size,
                pixels_per_meter=self._semantic_map_pixel_per_meter,
                polyline_thickness=int(self._semantic_map_pixel_per_meter / 11),
                # 1 when map_region_size == 2048, 2 for others
                layer=layer
            )
        else:
            logger.warning("Can not find map. Generate a square terrain")
            size = self._semantic_map_size * self._semantic_map_pixel_per_meter
            semantics = np.ones((size, size, 1), dtype=np.float32) * 0.2
        return semantics

    @staticmethod
    def make_render_state(engine, vert, frag):
        """
        Make a render state for specific camera
        Args:
            engine: BaseEngine
            vert: vert shader file name in shaders
            frag: frag shader file name in shaders

        Returns: RenderState

        """
        vert = AssetLoader.file_path("../shaders", vert)
        frag = AssetLoader.file_path("../shaders", frag)
        terrain_shader = Shader.load(Shader.SL_GLSL, vert, frag)

        dummy_np = NodePath("Dummy")
        dummy_np.setShader(terrain_shader)
        return dummy_np.getState()


# Some useful threads
# GeoMipTerrain:
# https://discourse.panda3d.org/t/texture-mapping-according-with-vertex-position-terrain-generation/28929
# https://github.com/someoneetc/Panda3dTerrainGenerator
# https://docs.panda3d.org/1.10/python/programming/texturing/texture-combine-modes
# texture splatting: https://discourse.panda3d.org/t/developing-a-city-builder-terrain-engine/7331
# https://discourse.panda3d.org/t/advice-about-geomipterrains/8346 optimize geomipterrain efficiency
#
# repeat texture
# # root = terrain.getRoot()
# # root.setTexScale(TextureStage.getDefault(), num_repeats_x, num_repeats_y)

# https://discourse.panda3d.org/t/geomipterrain-and-pnmimages/13728
# use shaderterrain with bullet: https://discourse.panda3d.org/t/getting-shaderterrainmesh-and-bulletheightfieldshape-to-match/27512
# shaderterrain+multi_texture: https://discourse.panda3d.org/t/alpha-texture-in-shaderterrainmesh-not-work/23715/3
# shaderTerrain height texture https://discourse.panda3d.org/t/how-get-height-of-shader-terrain-mesh-for-add-grass-tree-in-map/23964

# multi-thread pssm
# https://discourse.panda3d.org/t/parallel-split-shadow-mapping-using-pssmcamerarig/27457/12
