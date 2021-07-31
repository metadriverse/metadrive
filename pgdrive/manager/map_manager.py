import json
import logging
import os.path as osp

from pgdrive.component.map.base_map import BaseMap, MapGenerateMethod
from pgdrive.manager.base_manager import BaseManager
from pgdrive.utils import recursive_equal


class MapManager(BaseManager):
    """
    MapManager contains a list of maps
    """
    def __init__(self):
        super(MapManager, self).__init__()
        self.current_map = None

        # for pgmaps
        start_seed = self.engine.global_config["start_seed"]
        env_num = self.engine.global_config["environment_num"]
        self.restored_pg_map_configs = None
        self.pg_maps = {_seed: None for _seed in range(start_seed, start_seed + env_num)}

    def spawn_object(self, object_class, *args, **kwargs):
        if "random_seed" in kwargs:
            assert kwargs["random_seed"] == self.random_seed, "The random seed assigned is not same as map.seed"
            kwargs.pop("random_seed")
        map = super(MapManager, self).spawn_object(object_class, random_seed=self.random_seed, *args, **kwargs)
        self.pg_maps[map.random_seed] = map
        return map

    def load_all_maps_from_json(self, path):
        assert path.endswith(".json")
        assert osp.isfile(path)
        with open(path, "r") as f:
            config_and_data = json.load(f)
        global_config = self.engine.global_config
        start_seed = global_config["start_seed"]
        env_num = global_config["environment_num"]
        if recursive_equal(global_config["map_config"], config_and_data["map_config"]) \
                and set([i for i in range(start_seed, start_seed + env_num)]).issubset(
            set([int(v) for v in config_and_data["map_data"].keys()])):
            self.load_all_maps(config_and_data)
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

    def load_all_maps(self, data):
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
            map_config = {}
            map_config[BaseMap.GENERATE_TYPE] = MapGenerateMethod.PG_MAP_FILE
            map_config[BaseMap.GENERATE_CONFIG] = config
            self.restored_pg_map_configs[seed] = map_config

    def load_map(self, map):
        map.load_to_world()
        self.current_map = map

    def unload_map(self, map):
        map.unload_from_world()
        self.current_map = None

    def destroy(self):
        self.pg_maps = None
        self.restored_pg_map_configs = None
        super(MapManager, self).destroy()
