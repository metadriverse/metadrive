import copy

from metadrive.component.map.scenario_map import ScenarioMap
from metadrive.constants import DEFAULT_AGENT
from metadrive.manager.base_manager import BaseManager
from metadrive.scenario.parse_object_state import parse_full_trajectory, parse_object_state, get_idm_route
from metadrive.engine.logger import get_logger, set_log_level

logger = get_logger()


class ScenarioMapManager(BaseManager):
    PRIORITY = 0  # Map update has the most high priority
    DEFAULT_DATA_BUFFER_SIZE = 200

    def __init__(self):
        super(ScenarioMapManager, self).__init__()
        self.store_map = self.engine.global_config.get("store_map", False)
        self.current_map = None
        self._no_map = self.engine.global_config["no_map"]
        self.map_num = self.engine.global_config["num_scenarios"]
        self.start_scenario_index = self.engine.global_config["start_scenario_index"]
        self._stored_maps = {
            i: None
            for i in range(self.start_scenario_index, self.start_scenario_index + self.map_num)
        }

        # we put the route searching function here
        self.sdc_start_point = None
        self.sdc_destinations = []
        self.sdc_dest_point = None
        self.current_sdc_route = None

    def reset(self):
        if not self._no_map:
            seed = self.engine.global_random_seed
            assert self.start_scenario_index <= seed < self.start_scenario_index + self.map_num

            self.current_sdc_route = None
            self.sdc_dest_point = None

            if self._stored_maps[seed] is None:
                m_data = self.engine.data_manager.get_scenario(seed, should_copy=False)["map_features"]
                new_map = ScenarioMap(map_index=seed, map_data=m_data)
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

        init_state = parse_object_state(sdc_track, 0, check_last_state=False, include_z_position=True)

        # PZH: There is a wierd bug in the nuscene's source data, the width and length of the object is not consistent.
        # Maybe this should be handle in ScenarioNet. But for now, we have to handle it here.
        # As a workaround, we swap the width and length if the width is larger than length.
        if data["version"].startswith("nuscenesv1.0") or data["metadata"]["dataset"] == "nuscenes":
            if init_state["width"] > init_state["length"]:
                init_state["width"], init_state["length"] = init_state["length"], init_state["width"]

        if max(init_state["width"], init_state["length"]) > 2 and (init_state["width"] > init_state["length"]):
            logger.warning(
                "The width of the object {} is larger than length {}. Are you sure?".format(
                    init_state["width"], init_state["length"]
                )
            )

        last_state = parse_object_state(sdc_track, -1, check_last_state=True)
        init_position = init_state["position"]

        # Add a fake Z axis so that the object will not fall from the sky.
        init_position[-1] = 0

        init_yaw = init_state["heading"]
        last_position = last_state["position"]
        last_yaw = last_state["heading"]

        self.current_sdc_route = get_idm_route(sdc_traj)
        self.sdc_start_point = copy.deepcopy(init_position)
        self.sdc_dest_point = copy.deepcopy(last_position)

        self.engine.global_config.update(
            copy.deepcopy(
                dict(
                    agent_configs={
                        DEFAULT_AGENT: dict(
                            spawn_position_heading=(list(init_position), init_yaw),
                            spawn_velocity=init_state["velocity"],
                            width=init_state["width"],
                            length=init_state["length"],
                            height=init_state["height"],
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
        self.clear_stored_maps()
        self._stored_maps = None
        self.current_map = None

        self.sdc_start_point = None
        self.sdc_destinations = []
        self.sdc_dest_point = None
        self.current_sdc_route = None

        super(ScenarioMapManager, self).destroy()

    def before_reset(self):
        # remove map from world before adding
        if self.current_map is not None:
            self.unload_map(self.current_map)

        self.sdc_start_point = None
        self.sdc_destinations = []
        self.sdc_dest_point = None
        self.current_sdc_route = None
        self.current_map = None

    def clear_stored_maps(self):
        for m in self._stored_maps.values():
            if m is not None:
                m.detach_from_world()
                m.destroy()
        self._stored_maps = {
            i: None
            for i in range(self.start_scenario_index, self.start_scenario_index + self.map_num)
        }

    @property
    def num_stored_maps(self):
        return sum([1 if m is not None else 0 for m in self.engine.map_manager._stored_maps.values()])
