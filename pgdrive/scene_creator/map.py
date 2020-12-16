import copy
import json
import logging
import os
from typing import List

import numpy as np
import pygame
from panda3d.bullet import BulletWorld
from panda3d.core import NodePath

from pgdrive.pg_config import PgConfig
from pgdrive.pg_config.pg_blocks import PgBlock
from pgdrive.scene_creator.algorithm.BIG import BIG, BigGenerateMethod
from pgdrive.scene_creator.blocks.block import Block
from pgdrive.scene_creator.road.road_network import RoadNetwork
from pgdrive.utils.asset_loader import AssetLoader
from pgdrive.world.pg_world import PgWorld


class MapGenerateMethod:
    BIG_BLOCK_NUM = BigGenerateMethod.BLOCK_NUM
    BIG_BLOCK_SEQUENCE = BigGenerateMethod.BLOCK_SEQUENCE
    PG_MAP_FILE = "pg_map_file"


class Map:
    # only used to save and read maps
    FILE_SUFFIX = ".json"

    # define string in json and config
    SEED = "seed"
    LANE_WIDTH = "lane_width"
    LANE_NUM = "lane_num"
    BLOCK_ID = "id"
    BLOCK_SEQUENCE = "block_sequence"
    PRE_BLOCK_SOCKET_INDEX = "pre_block_socket_index"

    # generate_method
    GENERATE_PARA = "config"
    GENERATE_METHOD = "type"

    # draw with pygame, film size
    DRAW_LEN = 1024  # pix

    def __init__(self, pg_world: PgWorld, big_config: dict = None):
        """
        Map can be stored and recover to save time when we access the map encountered before
        """
        self.film_size = (self.DRAW_LEN, self.DRAW_LEN)
        parent_node_path, pg_physics_world = pg_world.worldNP, pg_world.physics_world
        self.config = self.default_config()
        if big_config:
            self.config.update(big_config)
        self.lane_width = self.config[self.LANE_WIDTH]
        self.lane_num = self.config[self.LANE_NUM]
        self.random_seed = self.config[self.SEED]
        self.road_network = RoadNetwork()
        self.blocks = []
        generate_type = self.config[self.GENERATE_METHOD]
        if generate_type == BigGenerateMethod.BLOCK_NUM or generate_type == BigGenerateMethod.BLOCK_SEQUENCE:
            self._big_generate(parent_node_path, pg_physics_world)

        elif generate_type == MapGenerateMethod.PG_MAP_FILE:
            # other config such as lane width, num and seed will be valid, since they will be read from file
            blocks_config = self.read_map(self.config[self.GENERATE_PARA])
            self._config_generate(blocks_config, parent_node_path, pg_physics_world)
        else:
            raise ValueError("Map can not be created by {}".format(generate_type))

        #  a trick to optimize performance
        self.road_network.update_indices()
        self.road_network.build_helper()
        self._load_to_highway_render(pg_world)

    @staticmethod
    def default_config():
        return PgConfig(
            {
                Map.GENERATE_METHOD: MapGenerateMethod.BIG_BLOCK_NUM,
                Map.GENERATE_PARA: None,  # it can be a file path / block num / block ID sequence
                Map.LANE_WIDTH: 3.5,
                Map.LANE_NUM: 3,
                Map.SEED: 10
            }
        )

    def _big_generate(self, parent_node_path: NodePath, pg_physics_world: BulletWorld):
        big_map = BIG(
            self.lane_num, self.lane_width, self.road_network, parent_node_path, pg_physics_world, self.random_seed
        )
        big_map.generate(self.config[self.GENERATE_METHOD], self.config[self.GENERATE_PARA])
        self.blocks = big_map.blocks

    def _config_generate(self, blocks_config: List, parent_node_path: NodePath, pg_physics_world: BulletWorld):
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"
        from pgdrive.scene_creator.blocks.first_block import FirstBlock
        last_block = FirstBlock(
            self.road_network, self.lane_width, self.lane_num, parent_node_path, pg_physics_world, 1
        )
        self.blocks.append(last_block)
        for block_index, b in enumerate(blocks_config[1:], 1):
            block_type = PgBlock.get_block(b.pop(self.BLOCK_ID))
            pre_block_socket_index = b.pop(self.PRE_BLOCK_SOCKET_INDEX)
            last_block = block_type(
                block_index, last_block.get_socket(pre_block_socket_index), self.road_network, self.random_seed
            )
            last_block.construct_from_config(b, parent_node_path, pg_physics_world)
            self.blocks.append(last_block)

    def load_to_pg_world(self, pg_world: PgWorld):
        parent_node_path, pg_physics_world = pg_world.worldNP, pg_world.physics_world
        for block in self.blocks:
            block.attach_to_pg_world(parent_node_path, pg_physics_world)
        self._load_to_highway_render(pg_world)

    def _load_to_highway_render(self, pg_world: PgWorld):
        if pg_world.pg_config["highway_render"]:
            pg_world.highway_render.set_map(self)

    def unload_from_pg_world(self, pg_world: PgWorld):
        for block in self.blocks:
            block.detach_from_pg_world(pg_world.physics_world)

    def destroy_map(self, pg_physics_world: BulletWorld):
        for block in self.blocks:
            block.destroy(pg_physics_world)

    def save_map(self):
        """
        This func will generate a json file named 'map_name.json', in 'save_dir'
        """
        assert self.blocks is not None and len(self.blocks) > 0, "Please generate Map before saving it"
        import numpy as np
        map_config = []
        for b in self.blocks:
            assert isinstance(b, Block), "None Block type can not be saved to json file"
            b_config = b.get_config()
            json_config = {}
            for k, v in b_config._config.items():
                json_config[k] = v.tolist()[0] if isinstance(v, np.ndarray) else v
            json_config[self.BLOCK_ID] = b.ID
            json_config[self.PRE_BLOCK_SOCKET_INDEX] = b.pre_block_socket_index
            map_config.append(json_config)

        saved_data = copy.deepcopy(
            {
                self.SEED: self.random_seed,
                self.LANE_NUM: self.lane_num,
                self.LANE_WIDTH: self.lane_width,
                self.BLOCK_SEQUENCE: map_config
            }
        )
        return saved_data

    def save_map_to_json(self, map_name: str, save_dir: str = os.path.dirname(__file__)):
        data = self.save_map()
        with open(AssetLoader.file_path(save_dir, map_name + self.FILE_SUFFIX), 'w') as outfile:
            json.dump(data, outfile)

    def read_map(self, map_config: dict):
        """
        Load the map from a dict
        """
        self.config[self.LANE_NUM] = map_config[self.LANE_NUM]
        self.config[self.LANE_WIDTH] = map_config[self.LANE_WIDTH]
        self.config[self.SEED] = map_config[self.SEED]
        blocks_config = map_config[self.BLOCK_SEQUENCE]

        # update the property
        self.lane_width = self.config[self.LANE_WIDTH]
        self.lane_num = self.config[self.LANE_NUM]
        self.random_seed = self.config[self.SEED]
        return blocks_config

    def read_map_from_json(self, map_file_path: str):
        """
        Create map from a .json file, read it to map config and update default properties
        """
        with open(map_file_path, "r") as map_file:
            map_config = json.load(map_file)
            ret = self.read_map(map_config)
        return ret

    def draw_map_image_on_surface(self) -> pygame.Surface:
        from pgdrive.world.highway_render.highway_render import LaneGraphics
        from pgdrive.world.highway_render.world_surface import WorldSurface
        surface = WorldSurface(self.film_size, 0, pygame.Surface(self.film_size))
        b_box = self.get_map_bound_box(self.road_network)
        x_len = b_box[1] - b_box[0]
        y_len = b_box[3] - b_box[2]
        max_len = max(x_len, y_len)
        # scaling and center can be easily found by bounding box
        scaling = self.film_size[1] / max_len - 0.1
        surface.scaling = scaling
        centering_pos = ((b_box[0] + b_box[1]) / 2, (b_box[2] + b_box[3]) / 2)
        surface.move_display_window_to(centering_pos)
        for _from in self.road_network.graph.keys():
            for _to in self.road_network.graph[_from].keys():
                for l in self.road_network.graph[_from][_to]:
                    LaneGraphics.simple_draw(l, surface)
        return surface

    @staticmethod
    def get_map_bound_box(road_network):
        from pgdrive.utils.math_utils import get_road_bound_box
        res_x_max = -np.inf
        res_x_min = np.inf
        res_y_min = np.inf
        res_y_max = -np.inf
        for _from, to_dict in road_network.graph.items():
            for _to, lanes in to_dict.items():
                x_max, x_min, y_max, y_min = get_road_bound_box(lanes)
                res_x_max = max(res_x_max, x_max)
                res_x_min = min(res_x_min, x_min)
                res_y_max = max(res_y_max, y_max)
                res_y_min = min(res_y_min, y_min)
        return res_x_min, res_x_max, res_y_min, res_y_max

    def get_map_image_array(self, fill_hole=True, only_black_white=True) -> np.ndarray:
        surface = self.draw_map_image_on_surface()
        if fill_hole:
            surface = self.fill_hole(surface)
        if only_black_white:
            return np.clip(pygame.surfarray.pixels_red(surface), 0.0, 1.0)
        return pygame.surfarray.array3d(surface)

    def save_map_image(self, fill_hole=True):
        surface = self.draw_map_image_on_surface()
        if fill_hole:
            surface = self.fill_hole(surface)
        pygame.image.save(surface, "map_{}.png".format(self.random_seed))

    @staticmethod
    def fill_hole(surface: pygame.Surface):
        res_surface = surface.copy()
        for i in range(surface.get_height()):
            for j in range(surface.get_width()):
                pix = surface.get_at((i, j))
                if pix == (255, 255, 255, 255):
                    continue
                count = 0
                for k_1 in [-1, 0, 1]:
                    for k_2 in [-1, 0, 1]:
                        if k_1 == 0 and k_2 == 0:
                            continue
                        if 0 < i + k_1 < surface.get_height() and 0 < j + k_2 < surface.get_width():
                            pix = surface.get_at((i + k_1, j + k_2))
                            if pix == (255, 255, 255, 255):
                                count += 1
                        else:
                            count += 1
                    if count >= 3:
                        res_surface.set_at((i, j), (255, 255, 255, 255))
                        break
        return res_surface

    def __del__(self):
        describe = self.random_seed if self.random_seed is not None else "custom"
        logging.debug("Scene {} is destroyed".format(describe))
