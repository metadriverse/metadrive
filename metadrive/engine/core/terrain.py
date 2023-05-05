# import numpy

import math

import numpy as np
from panda3d.bullet import BulletRigidBodyNode, BulletPlaneShape
from panda3d.bullet import ZUp, BulletHeightfieldShape
from panda3d.core import SamplerState, PNMImage, CardMaker, LQuaternionf
from panda3d.core import Vec3, ShaderTerrainMesh, Texture, TextureStage

from metadrive.base_class.base_object import BaseObject
from metadrive.constants import CamMask
from metadrive.constants import MetaDriveType, CollisionGroup
from metadrive.engine.asset_loader import AssetLoader
from metadrive.libs.diamond_square import diamond_square


class Terrain(BaseObject):
    COLLISION_MASK = CollisionGroup.Terrain
    HEIGHT = 0.0
    PROBE_HEIGHT = 600
    PROBE_SIZE = 1024

    def __init__(self, show_terrain, engine):
        use_render_pipeline = engine.use_render_pipeline
        super(Terrain, self).__init__(random_seed=0)

        # collision mesh
        self.simple_terrain_collision_mesh = None  # a flat collision shape
        self.terrain_collision_mesh = None  # a 3d mesh

        # visualization mesh feature
        heightfield_image_size = 4096  # fixed image size 4096*4096
        self._height_scale = engine.global_config["height_scale"]  # [m]
        self._drivable_region_extension = engine.global_config["drivable_region_extension"]  # [m] road marin
        self._terrain_size = 2048  # [m]
        self._semantic_map_size = 512  # [m] it should include the whole map. Otherwise, road will have no texture!
        self._semantic_map_pixel_per_meter = 22  # [m] how many pixels per meter
        # pre calculate some variables
        self._downsample_rate = int(heightfield_image_size / self._terrain_size)  # downsample to 2048 m
        self._elevation_texture_ratio = self._terrain_size / self._semantic_map_size  # for shader

        self._mesh_terrain = None
        self._mesh_terrain_height = None
        self._mesh_terrain_node = None
        self._terrain_shader_set = False  # only set once
        self.probe = None

        if self.render and show_terrain:
            if engine.use_render_pipeline:
                self._load_mesh_terrain_textures()
                self._mesh_terrain_node = ShaderTerrainMesh()
                # disable env probe as some vehicle models may break it
                # self.probe = engine.render_pipeline.add_environment_probe()
                # self.probe.set_pos(0, 0, self.PROBE_HEIGHT)
                # self.probe.set_scale(self.PROBE_SIZE * 2, self.PROBE_SIZE * 2, 1000)
            else:
                self._generate_card_terrain()

    def _load_mesh_terrain_textures(self, anisotropic_degree=16, minfilter=SamplerState.FT_linear_mipmap_linear):
        """
        Only maintain one copy of all asset
        :param anisotropic_degree: https://docs.panda3d.org/1.10/python/programming/texturing/texture-filter-types
        :param minfilter: https://docs.panda3d.org/1.10/python/programming/texturing/texture-filter-types
        :return: None
        """
        # texture stage
        self.ts_color = TextureStage("color")
        self.ts_normal = TextureStage("normal")
        self.ts_normal.setMode(TextureStage.M_normal)

        # grass
        self.grass_tex = self.loader.loadTexture(
            AssetLoader.file_path("textures", "grass2", "grass_path_2_diff_1k.png")
        )
        self.grass_normal = self.loader.loadTexture(
            AssetLoader.file_path("textures", "grass2", "grass_path_2_nor_gl_1k.png")
        )
        self.grass_rough = self.loader.loadTexture(
            AssetLoader.file_path("textures", "grass2", "grass_path_2_rough_1k.png")
        )

        v_wrap = Texture.WMRepeat
        u_warp = Texture.WMMirror
        filter_type = Texture.FTLinearMipmapLinear
        anisotropic_degree = 16
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

        v_wrap = Texture.WMRepeat
        u_warp = Texture.WMMirror
        filter_type = Texture.FTLinearMipmapLinear
        anisotropic_degree = 16
        for tex in [self.rock_tex, self.rock_normal, self.rock_rough]:
            tex.set_wrap_u(u_warp)
            tex.set_wrap_v(v_wrap)
            tex.setMinfilter(filter_type)
            tex.setMagfilter(filter_type)
            tex.setAnisotropicDegree(anisotropic_degree)

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

    # @time_me
    def _generate_mesh_vis_terrain(
        self,
        size,
        heightfield: Texture,
        attribute_tex: Texture,
        target_triangle_width=10,
        height_scale=100,
        height_offset=0.,
        engine=None,
    ):
        """
        Given a height field map to generate terrain and an attribute_tex to texture terrain, so we can get road/grass
        pixels_per_meter is determined by heightfield.size/size
        lane line and so on.
        :param size: [m] this terrain of the generate terrain
        :param heightfield: terrain heightfield. It should be a 16-bit png and have a quadratic size of a power of two.
        :param attribute_tex: doing texture splatting. r,g,b,a represent: grass/road/lane_line ratio respectively
        :param target_triangle_width: For a value of 10.0 for example, the terrain will attempt to make every triangle 10 pixels wide on screen.
        :param height_scale: Scale the height of mountain.
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

        # Generate the terrain
        self._mesh_terrain_node.generate()
        self._mesh_terrain = self.origin.attach_new_node(self._mesh_terrain_node)
        self._set_terrain_shader(engine, attribute_tex)

        # Attach the terrain to the main scene and set its scale. With no scale
        # set, the terrain ranges from (0, 0, 0) to (1, 1, 1)
        self._mesh_terrain.set_scale(size, size, height_scale)
        self._mesh_terrain_height = height_offset
        self._mesh_terrain.set_pos(-size / 2, -size / 2, self._mesh_terrain_height)

    def _set_terrain_shader(self, engine, attribute_tex):
        """
        Set a shader on the terrain. The ShaderTerrainMesh only works with an applied shader. You can use the shaders
        used here in your own application.

        Note: you have to make sure you modified the terrain_effect.yaml and vert.glsl/frag.glsl together, as they are
        made for different render pipeline. We expect the same terrain generated from different pipelines.
        """
        if engine.use_render_pipeline and not self._terrain_shader_set:
            engine.render_pipeline.reload_shaders()
            terrain_effect = AssetLoader.file_path("effect", "terrain_effect.yaml")
            engine.render_pipeline.set_effect(self._mesh_terrain, terrain_effect, {}, 100)
            # # height
            self._mesh_terrain.set_shader_input("height_scale", self._height_scale)

            # grass
            self._mesh_terrain.set_shader_input("grass_tex", self.grass_tex)
            self._mesh_terrain.set_shader_input("grass_normal", self.grass_normal)
            self._mesh_terrain.set_shader_input("grass_rough", self.grass_rough)

            # road
            self._mesh_terrain.set_shader_input("rock_tex", self.rock_tex)
            self._mesh_terrain.set_shader_input("rock_normal", self.rock_normal)
            self._mesh_terrain.set_shader_input("rock_rough", self.rock_rough)

            self._mesh_terrain.set_shader_input("road_tex", self.road_texture)
            self._mesh_terrain.set_shader_input("yellow_tex", self.yellow_lane_line)
            self._mesh_terrain.set_shader_input("white_tex", self.white_lane_line)
            self._mesh_terrain.set_shader_input("road_normal", self.road_texture_normal)
            self._mesh_terrain.set_shader_input("road_rough", self.road_texture_rough)
            self._mesh_terrain.set_shader_input("elevation_texture_ratio", self._elevation_texture_ratio)
            self._terrain_shader_set = True
        self._mesh_terrain.set_shader_input("attribute_tex", attribute_tex)

    def reset(self, center_position):
        """
        Update terrain according to current map
        """
        assert self.engine is not None, "Can not call this without initializing engine"

        # if not self.use_render_pipeline and self.simple_terrain_collision_mesh is None:
        # TODO: I disabled online terrain collision mesh generation now, consider enabling it in the future
        if self.simple_terrain_collision_mesh is None:
            self.detach_from_world(self.engine.physics_world)
            # If no render pipeline, we can only have 2d terrain. It will only be generated for once.
            shape = BulletPlaneShape(Vec3(0, 0, 1), -0.05)
            node = BulletRigidBodyNode(MetaDriveType.GROUND)
            node.setFriction(.9)
            node.addShape(shape)

            node.setIntoCollideMask(self.COLLISION_MASK)
            self.dynamic_nodes.append(node)

            self.simple_terrain_collision_mesh = self.origin.attachNewNode(node)
            self._node_path_list.append(np)
            self.attach_to_world(self.engine.render, self.engine.physics_world)

        # elif self.use_render_pipeline:
        if self.use_render_pipeline:
            self.detach_from_world(self.engine.physics_world)
            assert self.engine.current_map is not None, "Can not find current map"
            semantics = self.engine.current_map.get_semantic_map(
                size=self._semantic_map_size,
                pixels_per_meter=self._semantic_map_pixel_per_meter,
                polyline_thickness=int(1024 / self._semantic_map_size),
                layer=["lane", "lane_line"]
            )
            semantics = semantics.astype(np.float32)
            semantic_tex = Texture()
            semantic_tex.setup2dTexture(*semantics.shape[:2], Texture.TFloat, Texture.FRgba)
            semantic_tex.setRamImage(semantics)

            # we will downsmaple the precision after this
            heightfield_tex = self.loader.loadTexture(AssetLoader.file_path("textures", "terrain", "heightfield.png"))
            heightfield_img = np.frombuffer(heightfield_tex.getRamImage().getData(), dtype=np.uint16)
            heightfield_img = heightfield_img.reshape((heightfield_tex.getYSize(), heightfield_tex.getXSize(), 1))
            drivable_region = self.engine.current_map.get_height_map(
                self._terrain_size, self._downsample_rate, self._drivable_region_extension
            )
            drivable_region_height = np.mean(heightfield_img[np.where(drivable_region)]).astype(np.uint16)
            heightfield_img = np.where(drivable_region, drivable_region_height, heightfield_img)

            # set to zero height
            heightfield_img -= drivable_region_height

            # down sample
            heightfield_img = np.array(heightfield_img[::self._downsample_rate, ::self._downsample_rate])
            heightfield = heightfield_img

            heightfield_tex = Texture()
            heightfield_tex.setup2dTexture(*heightfield.shape[:2], Texture.TShort, Texture.FLuminance)
            heightfield_tex.setRamImage(heightfield)

            # # update collision every time!
            # TODO: I disabled online terrain collision mesh generation now, consider enabling it in the future
            # self._generate_collision_mesh(heightfield_img, self.height_scale)
            self._generate_mesh_vis_terrain(
                self._terrain_size, heightfield_tex, semantic_tex, height_scale=self._height_scale, height_offset=0
            )
            self.attach_to_world(self.engine.render, self.engine.physics_world)

        self.set_position(center_position)

    def _generate_collision_mesh(self, heightfield_img, height_scale):
        # TODO we can do some optimization here, only update some regions
        # clear previous mesh
        self.dynamic_nodes.clear()
        mesh = np.zeros([heightfield_img.shape[0] + 1, heightfield_img.shape[1] + 1, 1])
        mesh[:heightfield_img.shape[0], :heightfield_img.shape[1]] = heightfield_img
        mesh = mesh.astype(np.uint16)

        heightfield_tex = Texture()
        heightfield_tex.setup2dTexture(*mesh.shape[:2], Texture.TShort, Texture.FLuminance)
        heightfield_tex.setRamImage(mesh)

        p = PNMImage()
        heightfield_tex.store(p)

        shape = BulletHeightfieldShape(p, height_scale, ZUp)
        shape.setUseDiamondSubdivision(True)

        node = BulletRigidBodyNode(MetaDriveType.GROUND)
        node.setFriction(.9)
        node.addShape(shape)

        node.setIntoCollideMask(CollisionGroup.Terrain)
        self.dynamic_nodes.append(node)

        self.terrain_collision_mesh = self.origin.attachNewNode(node)
        self._node_path_list.append(np)

    def set_position(self, position, height=None):
        if self.render:
            if self.use_render_pipeline:
                pos = (self._mesh_terrain.get_pos()[0] + position[0], self._mesh_terrain.get_pos()[1] + position[1])
                self._mesh_terrain.set_pos(*pos, self._mesh_terrain_height)
                if self.probe is not None:
                    self.probe.set_pos(*pos, self.PROBE_HEIGHT)
            else:
                super(Terrain, self).set_position(position, height)

    def _generate_card_terrain(self):
        self.origin.hide(CamMask.MiniMap | CamMask.Shadow | CamMask.DepthCam | CamMask.ScreenshotCam)
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

    def _make_random_terrain(self, texture_size, terrain_size, heightfield):
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
