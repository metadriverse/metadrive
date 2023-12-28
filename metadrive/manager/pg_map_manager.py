import pickle

from tqdm import tqdm

from metadrive.component.map.pg_map import PGMap, MapGenerateMethod
from metadrive.manager.base_manager import BaseManager
from metadrive.utils.utils import get_time_str


class PGMapManager(BaseManager):
    """
    MapManager contains a list of PGmaps
    """
    PRIORITY = 0  # Map update has the most high priority

    def __init__(self):
        super(PGMapManager, self).__init__()
        self.current_map = None

        # for pgmaps
        start_seed = self.start_seed = self.engine.global_config["start_seed"]
        env_num = self.env_num = self.engine.global_config["num_scenarios"]
        self.maps = {_seed: None for _seed in range(start_seed, start_seed + env_num)}

    def spawn_object(self, object_class, *args, **kwargs):
        # Note: Map instance should not be reused / recycled.
        map = self.engine.spawn_object(object_class, auto_fill_random_seed=False, force_spawn=True, *args, **kwargs)
        self.engine._spawned_objects.pop(map.id)
        return map

    def load_map(self, map):
        map.attach_to_world()
        self.current_map = map

    def unload_map(self, map):
        map.detach_from_world()
        self.current_map = None
        if not self.engine.global_config["store_map"]:
            map.destroy()

    def destroy(self):
        super(PGMapManager, self).destroy()
        self.clear_stored_maps()
        self.maps = None

    def before_reset(self):
        # remove map from world before adding
        if self.current_map is not None:
            map = self.current_map
            self.unload_map(map)

    def reset(self):
        config = self.engine.global_config.copy()
        current_seed = self.engine.global_seed

        if self.maps[current_seed] is None:
            map_config = config["map_config"]
            map_config.update({"seed": current_seed})
            map_config = self.add_random_to_map(map_config)
            map = self.spawn_object(PGMap, map_config=map_config, random_seed=None)
            self.current_map = map
            if self.engine.global_config["store_map"]:
                self.maps[current_seed] = map
        else:
            map = self.maps[current_seed]
            self.load_map(map)

    def add_random_to_map(self, map_config):
        if self.engine.global_config["random_lane_width"]:
            map_config[PGMap.LANE_WIDTH
                       ] = self.np_random.rand() * (PGMap.MAX_LANE_WIDTH - PGMap.MIN_LANE_WIDTH) + PGMap.MIN_LANE_WIDTH
        if self.engine.global_config["random_lane_num"]:
            map_config[PGMap.LANE_NUM] = self.np_random.randint(PGMap.MIN_LANE_NUM, PGMap.MAX_LANE_NUM + 1)
        return map_config

    def generate_all_maps(self):
        """
        Call this function to generate all maps before using them
        """
        for seed in tqdm(self.maps.keys(), desc="Generate maps"):
            config = self.engine.global_config.copy()
            current_seed = seed
            self.engine.seed(seed)
            if self.maps[current_seed] is None:
                map_config = config["map_config"]
                map_config.update({"seed": current_seed})
                map_config = self.add_random_to_map(map_config)
                map = self.spawn_object(PGMap, map_config=map_config, random_seed=None)
                self.maps[current_seed] = map
                map.detach_from_world()

    def dump_all_maps(self, file_name=None):
        """
        Dump all maps. If some maps are not generated, we will generate it at first
        """
        if file_name is None:
            start_seed = self.engine.global_config["start_seed"]
            end_seed = start_seed + self.engine.global_config["num_scenarios"]
            file_name = "{}_{}_{}.json".format(start_seed, end_seed, get_time_str())
        self.generate_all_maps()
        ret = {}
        for seed, map in tqdm(self.maps.items(), desc="Dump maps"):
            ret[seed] = map.get_meta_data()
        with open(file_name, "wb+") as file:
            pickle.dump(ret, file)
        return ret

    def load_all_maps(self, file_name):
        if self.current_map is not None:
            self.unload_map(self.current_map)
        with open(file_name, "rb") as file:
            loaded_map_data = pickle.load(file)
        map_seeds = list(loaded_map_data.keys())
        start_seed = min(map_seeds)
        map_num = len(map_seeds)
        assert self.env_num == map_num and start_seed == self.engine.global_config[
            "start_seed"
        ], "The environment num and start seed in config: {}, {} must be the same as the env num and start seed: {}, {} in the loaded file".format(
            self.env_num, self.start_seed, map_num, start_seed
        )

        for i in tqdm(range(self.env_num), desc="Load maps"):
            loaded_seed = i + start_seed
            map_data = loaded_map_data[loaded_seed]
            block_sequence = map_data["block_sequence"]
            map_config = map_data["map_config"]
            map_config[PGMap.GENERATE_TYPE] = MapGenerateMethod.PG_MAP_FILE
            map_config[PGMap.GENERATE_CONFIG] = block_sequence
            map = self.spawn_object(PGMap, map_config=map_config, random_seed=None)
            self.maps[i + self.start_seed] = map
            map.detach_from_world()
        self.reset()
        return loaded_map_data

    def clear_stored_maps(self):
        """
        Clear all stored maps
        """
        for m in self.maps.values():
            if m is not None:
                m.detach_from_world()
                m.destroy()
        start_seed = self.start_seed = self.engine.global_config["start_seed"]
        env_num = self.env_num = self.engine.global_config["num_scenarios"]
        self.maps = {_seed: None for _seed in range(start_seed, start_seed + env_num)}
