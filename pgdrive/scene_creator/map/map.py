import copy
from pgdrive.utils.engine_utils import get_pgdrive_engine, set_global_random_seed
from pgdrive.scene_creator.road.road import Road
import logging
from pgdrive.utils.object import Object
import numpy as np
from pgdrive.scene_creator.algorithm.BIG import BigGenerateMethod
from pgdrive.scene_creator.blocks.block import Block
from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.scene_creator.road.road_network import RoadNetwork
from pgdrive.utils import PGConfig, import_pygame

pygame = import_pygame()


def parse_map_config(easy_map_config, new_map_config, default_config):
    assert isinstance(new_map_config, PGConfig)
    assert isinstance(default_config, PGConfig)

    # Return the user specified config if overwritten
    if not default_config["map_config"].is_identical(new_map_config):
        new_map_config = default_config["map_config"].copy(unchangeable=False).update(new_map_config)
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


class Map(Object):
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

    def __init__(self, map_config: dict = None, random_seed=None):
        """
        Map can be stored and recover to save time when we access the map encountered before
        """
        assert random_seed == map_config[
            self.SEED
        ], "Global seed {} should equal to seed in map config {}".format(random_seed, map_config[self.SEED])
        super(Map, self).__init__(random_seed=map_config[self.SEED])
        self.set_config(map_config)
        self.film_size = (self._config["draw_map_resolution"], self._config["draw_map_resolution"])
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
        self.unload_from_world()

    def _generate(self):
        """Key function! Please overwrite it!"""
        raise NotImplementedError("Please use child class like PGMap to replace Map!")

    def load_to_world(self):
        parent_node_path, pg_physics_world = self.pgdrive_engine.worldNP, self.pgdrive_engine.physics_world
        for block in self.blocks:
            block.attach_to_world(parent_node_path, pg_physics_world)

    def unload_from_world(self):
        for block in self.blocks:
            block.detach_from_world(self.pgdrive_engine.physics_world)

    def save_map(self):
        """
        Save the generated map to map file
        """
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
        self._config[self.SEED] = map_config[self.SEED]
        blocks_config = map_config[self.BLOCK_SEQUENCE]
        for b_id, b in enumerate(blocks_config):
            blocks_config[b_id] = {k: np.array(v) if isinstance(v, list) else v for k, v in b.items()}

        # update the property
        return blocks_config

    @property
    def num_blocks(self):
        return len(self.blocks)

    def destroy(self):
        for block in self.blocks:
            block.destroy()
        super(Map, self).destroy()

    def __del__(self):
        describe = self.random_seed if self.random_seed is not None else "custom"
        logging.debug("Scene {} is destroyed".format(describe))
