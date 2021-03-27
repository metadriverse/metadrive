import copy
import json
import logging
import os
from typing import List

import numpy as np
from panda3d.core import NodePath

from pgdrive.scene_creator.algorithm.BIG import BIG, BigGenerateMethod
from pgdrive.scene_creator.blocks.block import Block
from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.scene_creator.pg_blocks import PGBlock
from pgdrive.scene_creator.road.road_network import RoadNetwork
from pgdrive.utils import PGConfig, AssetLoader, import_pygame
from pgdrive.world.pg_physics_world import PGPhysicsWorld
from pgdrive.world.pg_world import PGWorld

pygame = import_pygame()


def parse_map_config(easy_map_config, original_map_config):
    assert isinstance(original_map_config, PGConfig)
    if isinstance(easy_map_config, int):
        original_map_config[Map.GENERATE_TYPE] = BigGenerateMethod.BLOCK_NUM
    elif isinstance(easy_map_config, str):
        original_map_config[Map.GENERATE_TYPE] = BigGenerateMethod.BLOCK_SEQUENCE
    else:
        raise ValueError(
            "Unkown easy map config: {} and original map config: {}".format(easy_map_config, original_map_config)
        )
    original_map_config[Map.GENERATE_CONFIG] = easy_map_config
    return original_map_config


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
    LANE_WIDTH_RAND_RANGE = "lane_width_rand_range"
    LANE_NUM = "lane_num"
    BLOCK_ID = "id"
    BLOCK_SEQUENCE = "block_sequence"
    PRE_BLOCK_SOCKET_INDEX = "pre_block_socket_index"

    # generate_method
    GENERATE_CONFIG = "config"
    GENERATE_TYPE = "type"

    def __init__(self, pg_world: PGWorld, map_config: dict = None):
        """
        Map can be stored and recover to save time when we access the map encountered before
        """
        self.config = PGConfig(map_config)
        self.film_size = (self.config["draw_map_resolution"], self.config["draw_map_resolution"])
        self.random_seed = self.config[self.SEED]
        self.road_network = RoadNetwork()
        self.blocks = []

        self._generate(pg_world)

        #  a trick to optimize performance
        self.road_network.after_init()

    def _generate(self, pg_world):
        """
        We can override this function to introduce other methods!
        """
        parent_node_path, pg_physics_world = pg_world.worldNP, pg_world.physics_world
        generate_type = self.config[self.GENERATE_TYPE]
        if generate_type == BigGenerateMethod.BLOCK_NUM or generate_type == BigGenerateMethod.BLOCK_SEQUENCE:
            self._big_generate(parent_node_path, pg_physics_world)

        elif generate_type == MapGenerateMethod.PG_MAP_FILE:
            # other config such as lane width, num and seed will be valid, since they will be read from file
            blocks_config = self.read_map(self.config[self.GENERATE_CONFIG])
            self._config_generate(blocks_config, parent_node_path, pg_physics_world)
        else:
            raise ValueError("Map can not be created by {}".format(generate_type))

    def _big_generate(self, parent_node_path: NodePath, pg_physics_world: PGPhysicsWorld):
        big_map = BIG(
            self.config[self.LANE_NUM], self.config[self.LANE_WIDTH], self.road_network, parent_node_path,
            pg_physics_world, self.random_seed
        )
        big_map.generate(self.config[self.GENERATE_TYPE], self.config[self.GENERATE_CONFIG])
        self.blocks = big_map.blocks

    def _config_generate(self, blocks_config: List, parent_node_path: NodePath, pg_physics_world: PGPhysicsWorld):
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"
        last_block = FirstBlock(
            self.road_network, self.config[self.LANE_WIDTH], self.config[self.LANE_NUM], parent_node_path,
            pg_physics_world, 1
        )
        self.blocks.append(last_block)
        for block_index, b in enumerate(blocks_config[1:], 1):
            block_type = PGBlock.get_block(b.pop(self.BLOCK_ID))
            pre_block_socket_index = b.pop(self.PRE_BLOCK_SOCKET_INDEX)
            last_block = block_type(
                block_index, last_block.get_socket(pre_block_socket_index), self.road_network, self.random_seed
            )
            last_block.construct_from_config(b, parent_node_path, pg_physics_world)
            self.blocks.append(last_block)

    def load_to_pg_world(self, pg_world: PGWorld):
        parent_node_path, pg_physics_world = pg_world.worldNP, pg_world.physics_world
        for block in self.blocks:
            block.attach_to_pg_world(parent_node_path, pg_physics_world)

    def unload_from_pg_world(self, pg_world: PGWorld):
        for block in self.blocks:
            block.detach_from_pg_world(pg_world.physics_world)

    def destroy(self, pg_world: PGWorld):
        for block in self.blocks:
            block.destroy(pg_world=pg_world)

    def save_map(self):
        assert self.blocks is not None and len(self.blocks) > 0, "Please generate Map before saving it"
        map_config = []
        for b in self.blocks:
            assert isinstance(b, Block), "None Set can not be saved to json file"
            b_config = b.get_config()
            json_config = b_config.get_dict()
            json_config[self.BLOCK_ID] = b.ID
            json_config[self.PRE_BLOCK_SOCKET_INDEX] = b.pre_block_socket_index
            map_config.append(json_config)

        saved_data = copy.deepcopy({self.SEED: self.random_seed, self.BLOCK_SEQUENCE: map_config})
        return saved_data

    def save_map_to_json(self, map_name: str, save_dir: str = os.path.dirname(__file__)):
        """
        This func will generate a json file named 'map_name.json', in 'save_dir'
        """
        data = self.save_map()
        with open(AssetLoader.file_path(save_dir, map_name + self.FILE_SUFFIX), 'w') as outfile:
            json.dump(data, outfile)

    def read_map(self, map_config: dict):
        """
        Load the map from a dict
        """
        self.config[self.SEED] = map_config[self.SEED]
        blocks_config = map_config[self.BLOCK_SEQUENCE]
        for b_id, b in enumerate(blocks_config):
            blocks_config[b_id] = {k: np.array(v) if isinstance(v, list) else v for k, v in b.items()}

        # update the property
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

    def __del__(self):
        describe = self.random_seed if self.random_seed is not None else "custom"
        logging.debug("Scene {} is destroyed".format(describe))

    @property
    def num_blocks(self):
        return len(self.blocks)
