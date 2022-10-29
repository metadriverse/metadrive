from metadrive.base_class.base_runnable import BaseRunnable
from metadrive.engine.engine_utils import get_engine, get_global_config
from metadrive.utils import import_pygame

pygame = import_pygame()


class BaseMap(BaseRunnable):
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

    # default lane parameter
    MAX_LANE_WIDTH = 4.5
    MIN_LANE_WIDTH = 3.0
    MAX_LANE_NUM = 3
    MIN_LANE_NUM = 2

    def __init__(self, map_config: dict = None, random_seed=None):
        """
        Map can be stored and recover to save time when we access the map encountered before
        """
        assert random_seed is None
        # assert random_seed == map_config[
        #     self.SEED
        # ], "Global seed {} should equal to seed in map config {}".format(random_seed, map_config[self.SEED])
        super(BaseMap, self).__init__(config=map_config)
        self.film_size = (get_global_config()["draw_map_resolution"], get_global_config()["draw_map_resolution"])
        self.road_network = self.road_network_type()

        # A flatten representation of blocks, might cause chaos in city-level generation.
        self.blocks = []

        # Generate map and insert blocks
        self.engine = get_engine()
        self._generate()
        assert self.blocks, "The generate methods does not fill blocks!"

        #  a trick to optimize performance
        self.spawn_roads = None
        self.detach_from_world()

    def _generate(self):
        """Key function! Please overwrite it! This func aims at fill the self.road_network adn self.blocks"""
        raise NotImplementedError("Please use child class like PGMap to replace Map!")

    def attach_to_world(self, parent_np=None, physics_world=None):
        parent_node_path, physics_world = self.engine.worldNP or parent_np, self.engine.physics_world or physics_world
        for block in self.blocks:
            block.attach_to_world(parent_node_path, physics_world)

    def detach_from_world(self, physics_world=None):
        for block in self.blocks:
            block.detach_from_world(self.engine.physics_world or physics_world)

    def get_meta_data(self):
        """
        Save the generated map to map file
        """
        raise NotImplementedError

    @property
    def num_blocks(self):
        return len(self.blocks)

    def destroy(self):
        self.detach_from_world()
        for block in self.blocks:
            block.destroy()
        self.blocks = None
        self.road_network.destroy()
        self.road_network = None
        self.spawn_roads = None
        super(BaseMap, self).destroy()

    @property
    def road_network_type(self):
        raise NotImplementedError

    def get_center_point(self):
        x_min, x_max, y_min, y_max = self.road_network.get_bounding_box()
        return (x_max + x_min) / 2, (y_max + y_min) / 2
