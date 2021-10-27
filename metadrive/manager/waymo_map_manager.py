import os

from metadrive.component.map.waymo_map import WaymoMap
from metadrive.manager.base_manager import BaseManager
from metadrive.utils.waymo_map_utils import read_waymo_data


class WaymoMapManager(BaseManager):
    PRIORITY = 0  # Map update has the most high priority

    def __init__(self):
        super(WaymoMapManager, self).__init__()
        self.current_map = None
        self.map_num = self.engine.global_config["map_num"]
        directory = self.engine.global_config["map_directory"]
        self._map_configs = {}
        for i in range(self.map_num):
            file_path = os.path.join(directory, "{}.pkl".format(i))
            data = read_waymo_data(file_path)
            self._map_configs[i] = data
        self.maps = {_seed: None for _seed in range(0, self.map_num)}

    def reset(self):
        seed = self.np_random.randint(0, self.map_num)
        if self.maps[seed] is None:
            map_config = self._map_configs[seed]
            map = self.spawn_object(WaymoMap, waymo_data=map_config)
            self.maps[seed] = map
        map = self.maps[seed]
        self.load_map(map)

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
        self.maps = None
        self.current_map = None
        self._map_configs = None
        super(WaymoMapManager, self).destroy()

    def before_reset(self):
        # remove map from world before adding
        if self.current_map is not None:
            self.unload_map(self.current_map)
