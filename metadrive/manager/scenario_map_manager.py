import copy

from metadrive.component.map.scenario_map import ScenarioMap
from metadrive.constants import DEFAULT_AGENT
from metadrive.manager.base_manager import BaseManager
from metadrive.scenario.parse_object_state import parse_full_trajectory, parse_object_state, get_idm_route


class ScenarioMapManager(BaseManager):
    PRIORITY = 0  # Map update has the most high priority
    DEFAULT_DATA_BUFFER_SIZE = 200

    def __init__(self):
        super(ScenarioMapManager, self).__init__()
        self.store_map = self.engine.global_config.get("store_map", False)
        self.current_map = None
        self.map_num = self.engine.global_config["num_scenarios"]
        self.start_scenario_index = self.engine.global_config["start_scenario_index"]
        self._stored_maps = {
            i: None
            for i in range(self.start_scenario_index, self.start_scenario_index + self.map_num)
        }

        # we put the route searching function here
        self.sdc_start = None
        self.sdc_destinations = []
        self.sdc_dest_point = None
        self.current_sdc_route = None

    def reset(self):
        seed = self.engine.global_random_seed
        assert self.start_scenario_index <= seed < self.start_scenario_index + self.map_num

        self.current_sdc_route = None
        self.sdc_dest_point = None

        if self._stored_maps[seed] is None:
            new_map = ScenarioMap(map_index=seed)
            if self.store_map:
                self._stored_maps[seed] = new_map
        else:
            new_map = self._stored_maps[seed]
        self.load_map(new_map)
        self.update_route()

    def update_route(self):
        data = self.engine.data_manager.current_scenario

        sdc_track = data.get_sdc_track()

        sdc_traj = parse_full_trajectory(sdc_track)

        init_state = parse_object_state(sdc_track, 0, check_last_state=False)
        last_state = parse_object_state(sdc_track, -1, check_last_state=True)
        init_position = init_state["position"]
        init_yaw = init_state["heading"]
        last_position = last_state["position"]
        last_yaw = last_state["heading"]

        self.current_sdc_route = get_idm_route(sdc_traj)

        self.sdc_dest_point = copy.deepcopy(last_position)

        self.engine.global_config.update(
            copy.deepcopy(
                dict(
                    target_vehicle_configs={
                        DEFAULT_AGENT: dict(
                            spawn_position_heading=(init_position, init_yaw), spawn_velocity=init_state["velocity"]
                        )
                    }
                )
            )
        )

    def filter_path(self, start_lanes, end_lanes):
        for start in start_lanes:
            for end in end_lanes:
                path = self.current_map.road_network.shortest_path(start[0].index, end[0].index)
                if len(path) > 0:
                    return (start[0].index, end[0].index)
        return None

    def spawn_object(self, object_class, *args, **kwargs):
        raise ValueError("Please create ScenarioMap instance directly without calling spawn_object function.")
        # map = self.engine.spawn_object(object_class, auto_fill_random_seed=False, *args, **kwargs)
        # self.spawned_objects[map.id] = map
        # return map

    def load_map(self, map):
        map.attach_to_world()
        self.current_map = map

    def unload_map(self, map):
        map.detach_from_world()
        self.current_map = None
        if not self.engine.global_config["store_map"]:
            map.destroy()
            assert len(self.spawned_objects) == 0

    def destroy(self):
        self.maps = None
        self.current_map = None

        self.sdc_start = None
        self.sdc_destinations = []
        self.sdc_dest_point = None
        self.current_sdc_route = None

        super(ScenarioMapManager, self).destroy()

    def before_reset(self):
        # remove map from world before adding
        if self.current_map is not None:
            self.unload_map(self.current_map)

        self.sdc_start = None
        self.sdc_destinations = []
        self.sdc_dest_point = None
        self.current_sdc_route = None
        self.current_map = None

    def clear_objects(self, *args, **kwargs):
        """
        As Map instance should not be recycled, we will forcefully destroy useless map instances.
        """
        return super(ScenarioMapManager, self).clear_objects(force_destroy=True, *args, **kwargs)
