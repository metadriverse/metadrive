import copy
import json
import logging
import os.path as osp

from metadrive.component.map.base_map import BaseMap, MapGenerateMethod
from metadrive.component.map.pg_map import PGMap
from metadrive.manager.base_manager import BaseManager
from metadrive.utils import recursive_equal


class MapManager(BaseManager):
    """
    MapManager contains a list of maps
    """
    PRIORITY = 0  # Map update has the most high priority

    def __init__(self):
        super(MapManager, self).__init__()
        self.current_map = None

        # for pgmaps
        start_seed = self.engine.global_config["start_seed"]
        env_num = self.engine.global_config["environment_num"]
        self.pg_maps = {_seed: None for _seed in range(start_seed, start_seed + env_num)}

    def spawn_object(self, object_class, *args, **kwargs):
        map = self.engine.spawn_object(object_class, auto_fill_random_seed=False, *args, **kwargs)
        return map

    def load_map(self, map):
        map.attach_to_world()
        self.current_map = map

    def unload_map(self, map):
        map.detach_from_world()
        self.current_map = None

    def destroy(self):
        self.pg_maps = None
        super(MapManager, self).destroy()

    def before_reset(self):
        # remove map from world before adding
        if self.current_map is not None:
            self.unload_map(self.current_map)

    def reset(self):
        # if episode_data is not None:
        #     # TODO restore/replay here
        #     # Since in episode data map data only contains one map, values()[0] is the map_parameters
        #     map_data = episode_data["map_data"].values()
        #     assert len(map_data) > 0, "Can not find map info in episode data"
        #     blocks_info = map_data[0]
        #
        #     map_config = copy.deepcopy(config["map_config"])
        #     # map_config[BaseMap.GENERATE_TYPE] = MapGenerateMethod.PG_MAP_FILE
        #     # map_config[BaseMap.GENERATE_CONFIG] = blocks_info
        #     map_config.update(map_data)
        #     self.spawn_object(PGMap, map_config=map_config)
        #     return
        config = self.engine.global_config.copy()
        current_seed = self.engine.global_seed

        if self.pg_maps[current_seed] is None:
            map_config = config["map_config"]
            map_config.update({"seed": current_seed})
            map_config = self.add_random_to_map(map_config)
            map = self.spawn_object(PGMap, map_config=map_config, random_seed=None)
            self.pg_maps[current_seed] = map
        map = self.pg_maps[current_seed]
        self.load_map(map)

    def add_random_to_map(self, map_config):
        if self.engine.global_config["random_lane_width"]:
            map_config[PGMap.LANE_WIDTH
                       ] = self.np_random.rand() * (PGMap.MAX_LANE_WIDTH - PGMap.MIN_LANE_WIDTH) + PGMap.MIN_LANE_WIDTH
        if self.engine.global_config["random_lane_num"]:
            map_config[PGMap.LANE_NUM] = self.np_random.randint(PGMap.MIN_LANE_NUM, PGMap.MAX_LANE_NUM + 1)
        return map_config
