import copy
from pgdrive.utils.engine_utils import get_pgdrive_engine
from pgdrive.scene_creator.road.road import Road
import logging
from typing import List

import numpy as np
from panda3d.core import NodePath
from pgdrive.scene_creator.algorithm.BIG import BIG, BigGenerateMethod
from pgdrive.scene_creator.blocks.block import Block
from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.scene_creator.pg_blocks import PGBlock
from pgdrive.scene_creator.road.road_network import RoadNetwork
from pgdrive.utils import PGConfig, import_pygame
from pgdrive.engine.world.pg_physics_world import PGPhysicsWorld

pygame = import_pygame()


def parse_map_config(easy_map_config, new_map_config, default_config):
    assert isinstance(new_map_config, PGConfig)
    assert isinstance(default_config, PGConfig)

    # Return the user specified config if overwritten
    if not default_config["map_config"].is_identical(new_map_config):
        new_map_config = default_config.copy(unchangeable=False).update(new_map_config)
        assert default_config["map"] == easy_map_config
        return new_map_config

    if isinstance(easy_map_config, int):
        new_map_config[Map.GENERATE_TYPE] = BigGenerateMethod.BLOCK_NUM
    elif isinstance(easy_map_config, str):
        new_map_config[Map.GENERATE_TYPE] = BigGenerateMethod.BLOCK_SEQUENCE
    else:
        raise ValueError(
            "Unkown easy map config: {} and original map config: {}".format(easy_map_config, new_map_config)
        )
    new_map_config[Map.GENERATE_CONFIG] = easy_map_config
    return new_map_config


class MapGenerateMethod:
    BIG_BLOCK_NUM = BigGenerateMethod.BLOCK_NUM
    BIG_BLOCK_SEQUENCE = BigGenerateMethod.BLOCK_SEQUENCE
    BIG_SINGLE_BLOCK = BigGenerateMethod.SINGLE_BLOCK
    PG_MAP_FILE = "pg_map_file"


class Map:
    """
    Base class for Map generation!
    """
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

    def __init__(self, map_config: dict = None):
        """
        Map can be stored and recover to save time when we access the map encountered before
        """
        self.config = PGConfig(map_config)
        self.film_size = (self.config["draw_map_resolution"], self.config["draw_map_resolution"])
        self.random_seed = self.config[self.SEED]
        self.road_network = RoadNetwork()

        # A flatten representation of blocks, might cause chaos in city-level generation.
        self.blocks = []

        # Generate map and insert blocks
        self.pgdrive_engine = get_pgdrive_engine()
        self._generate()
        assert self.blocks, "The generate methods does not fill blocks!"

        #  a trick to optimize performance
        self.road_network.after_init()
        self.spawn_roads = [Road(FirstBlock.NODE_2, FirstBlock.NODE_3)]

    def _generate(self):
        """Key function! Please overwrite it!"""
        raise NotImplementedError("Please use child class like PGMap to replace Map!")

    def load_to_pg_world(self):
        parent_node_path, pg_physics_world = self.pgdrive_engine.worldNP, self.pgdrive_engine.physics_world
        for block in self.blocks:
            block.attach_to_pg_world(parent_node_path, pg_physics_world)

    def unload_from_pg_world(self):
        for block in self.blocks:
            block.detach_from_pg_world(self.pgdrive_engine.physics_world)

    def destroy(self):
        for block in self.blocks:
            block.destroy()

    def save_map(self):
        assert self.blocks is not None and len(self.blocks) > 0, "Please generate Map before saving it"
        map_config = []
        for b in self.blocks:
            assert isinstance(b, Block), "None Set can not be saved to json file"
            b_config = b.get_config()
            json_config = b_config.get_serializable_dict()
            json_config[self.BLOCK_ID] = b.ID
            json_config[self.PRE_BLOCK_SOCKET_INDEX] = b.pre_block_socket_index
            map_config.append(json_config)

        saved_data = copy.deepcopy({self.SEED: self.random_seed, self.BLOCK_SEQUENCE: map_config})
        return saved_data

    def read_map(self, map_config: dict):
        """
        Load the map from a dict. Note that we don't provide a restore function in the base class.
        """
        self.config[self.SEED] = map_config[self.SEED]
        blocks_config = map_config[self.BLOCK_SEQUENCE]
        for b_id, b in enumerate(blocks_config):
            blocks_config[b_id] = {k: np.array(v) if isinstance(v, list) else v for k, v in b.items()}

        # update the property
        self.random_seed = self.config[self.SEED]
        return blocks_config

    @property
    def num_blocks(self):
        return len(self.blocks)

    def __del__(self):
        describe = self.random_seed if self.random_seed is not None else "custom"
        logging.debug("Scene {} is destroyed".format(describe))


class PGMap(Map):
    def _generate(self):
        """
        We can override this function to introduce other methods!
        """
        parent_node_path, pg_physics_world = self.pgdrive_engine.worldNP, self.pgdrive_engine.physics_world
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
            self.config[self.LANE_NUM],
            self.config[self.LANE_WIDTH],
            self.road_network,
            parent_node_path,
            pg_physics_world,
            self.random_seed,
            self.config["block_type_version"],
            exit_length=self.config["exit_length"]
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
            block_type = PGBlock.get_block(b.pop(self.BLOCK_ID), self.config["block_type_version"])
            pre_block_socket_index = b.pop(self.PRE_BLOCK_SOCKET_INDEX)
            last_block = block_type(
                block_index, last_block.get_socket(pre_block_socket_index), self.road_network, self.random_seed
            )
            last_block.construct_from_config(b, parent_node_path, pg_physics_world)
            self.blocks.append(last_block)
