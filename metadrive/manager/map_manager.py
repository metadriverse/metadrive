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

    def __init__(self, single_block=None):
        super(MapManager, self).__init__()
        self.current_map = None

        # for pgmaps
        start_seed = self.engine.global_config["start_seed"]
        env_num = self.engine.global_config["environment_num"]
        self.restored_pg_map_configs = None
        self.pg_maps = {_seed: None for _seed in range(start_seed, start_seed + env_num)}

        # TODO(pzh): clean this!
        self.single_block_class = single_block

    def spawn_object(self, object_class, *args, **kwargs):
        map = self.engine.spawn_object(object_class, auto_fill_random_seed=False, *args, **kwargs)
        return map

    def load_map(self, map):
        map.attach_to_world()
        self.current_map = map

    def unload_map(self, map):
        map.detach_from_world()
        self.current_map = None

    def read_all_maps_from_json(self, path):
        assert path.endswith(".json")
        assert osp.isfile(path), path
        with open(path, "r") as f:
            config_and_data = json.load(f)
        global_config = self.engine.global_config
        start_seed = global_config["start_seed"]
        env_num = global_config["environment_num"]
        if recursive_equal(global_config["map_config"], config_and_data["map_config"]) \
                and set([i for i in range(start_seed, start_seed + env_num)]).issubset(
            set([int(v) for v in config_and_data["map_data"].keys()])):
            self.read_all_maps(config_and_data)
            return True
        else:
            logging.warning(
                "Warning: The pre-generated maps is with config {}, but current environment's map "
                "config is {}.\nWe now fallback to BIG algorithm to generate map online!".format(
                    config_and_data["map_config"], global_config["map_config"]
                )
            )
            global_config["load_map_from_json"] = False  # Don't fall into this function again.
            return False

    def read_all_maps(self, data):
        assert isinstance(data, dict)
        assert set(data.keys()) == {"map_config", "map_data"}
        logging.info(
            "Restoring the maps from pre-generated file! "
            "We have {} maps in the file and restoring {} maps range from {} to {}".format(
                len(data["map_data"]), len(self.pg_maps.keys()), min(self.pg_maps.keys()), max(self.pg_maps.keys())
            )
        )

        maps_collection_config = data["map_config"]
        assert set(self.engine.global_config["map_config"].keys()) == set(maps_collection_config.keys())
        for k in self.engine.global_config["map_config"]:
            assert maps_collection_config[k] == self.engine.global_config["map_config"][k]
        self.restored_pg_map_configs = {}
        # for seed, map_dict in data["map_data"].items():
        for seed, config in data["map_data"].items():
            seed = int(seed)
            map_config = copy.deepcopy(maps_collection_config)
            map_config[BaseMap.GENERATE_TYPE] = MapGenerateMethod.PG_MAP_FILE
            map_config[BaseMap.GENERATE_CONFIG] = config

            # map_config.update(config)

            self.restored_pg_map_configs[seed] = map_config
            # self.restored_pg_map_configs[seed] = config

    def destroy(self):
        self.pg_maps = None
        self.restored_pg_map_configs = None
        super(MapManager, self).destroy()

    def update_map(self, config, current_seed, episode_data: dict = None, single_block_class=None, spawn_roads=None):
        # TODO(pzh): Remove the config as the input args.
        if episode_data is not None:
            # TODO restore/replay here
            # Since in episode data map data only contains one map, values()[0] is the map_parameters
            map_data = episode_data["map_data"].values()
            assert len(map_data) > 0, "Can not find map info in episode data"
            blocks_info = map_data[0]

            map_config = copy.deepcopy(config["map_config"])
            # map_config[BaseMap.GENERATE_TYPE] = MapGenerateMethod.PG_MAP_FILE
            # map_config[BaseMap.GENERATE_CONFIG] = blocks_info
            map_config.update(map_data)
            self.spawn_object(PGMap, map_config=map_config)
            return

        # Build single block for multi-agent system!
        if config["is_multi_agent"] and single_block_class is not None:
            assert single_block_class is not None
            assert spawn_roads is not None
            if self.current_map is None:
                new_map = self.spawn_object(single_block_class, map_config=config["map_config"], random_seed=None)
                self.load_map(new_map)
                self.current_map.spawn_roads = spawn_roads
            return

        # If we choose to load maps from json file.
        if config["load_map_from_json"] and self.current_map is None:
            assert config["map_file_path"]
            logging.info("Loading maps from: {}".format(config["map_file_path"]))
            self.read_all_maps_from_json(config["map_file_path"])

        # remove map from world before adding
        if self.current_map is not None:
            self.unload_map(self.current_map)

        if self.pg_maps[current_seed] is None:
            if config["load_map_from_json"]:
                map_config = self.restored_pg_map_configs.get(current_seed, None)
                assert map_config is not None
                logging.debug(
                    "We are spawning predefined map (seed {}). This is the config: {}".format(current_seed, map_config)
                )
            else:
                map_config = config["map_config"]
                map_config.update({"seed": current_seed})

                logging.debug(
                    "We are spawning new map (seed {}). This is the config: {}".format(current_seed, map_config)
                )
            map_config = self.add_random_to_map(map_config)
            map = self.spawn_object(PGMap, map_config=map_config, random_seed=None)
            self.pg_maps[current_seed] = map
        else:
            logging.debug("We are loading map from pg_maps (seed {}): {}".format(current_seed, len(self.pg_maps)))
            map = self.pg_maps[current_seed]
        # print("WE ARE LOADING MAP SEED {}.".format(current_seed))
        self.load_map(map)

    def add_random_to_map(self, map_config):
        if self.engine.global_config["random_lane_width"]:
            assert not self.engine.global_config["load_map_from_json"
                                                 ], "You are supposed to turn off the load_map_from_json"
            map_config[PGMap.LANE_WIDTH
                       ] = self.np_random.rand() * (PGMap.MAX_LANE_WIDTH - PGMap.MIN_LANE_WIDTH) + PGMap.MIN_LANE_WIDTH

        if self.engine.global_config["random_lane_num"]:
            assert not self.engine.global_config["load_map_from_json"
                                                 ], "You are supposed to turn off the load_map_from_json"
            map_config[PGMap.LANE_NUM] = self.np_random.randint(PGMap.MIN_LANE_NUM, PGMap.MAX_LANE_NUM+1)

        return map_config
