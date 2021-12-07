from metadrive.component.map.waymo_map import WaymoMap
from metadrive.manager.base_manager import BaseManager


class WaymoMapManager(BaseManager):
    PRIORITY = 0  # Map update has the most high priority

    def __init__(self):
        super(WaymoMapManager, self).__init__()
        self.current_map = None
        self.map_num = self.engine.global_config["case_num"]
        self.maps = {_seed: None for _seed in range(0, self.map_num)}

    def reset(self):
        seed = self.engine.global_random_seed
        if self.maps[seed] is None:
            map_config = self.engine.data_manager.cases[seed]
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
        super(WaymoMapManager, self).destroy()

    def before_reset(self):
        # remove map from world before adding
        if self.current_map is not None:
            self.unload_map(self.current_map)
