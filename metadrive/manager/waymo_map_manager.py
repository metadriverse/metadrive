import copy

from metadrive.component.lane.point_lane import PointLane
from metadrive.component.map.waymo_map import WaymoMap
from metadrive.constants import DEFAULT_AGENT
from metadrive.manager.base_manager import BaseManager
from metadrive.scenario.scenario_description import ScenarioDescription as SD
from metadrive.utils.waymo_utils.parse_object_state import parse_full_trajectory, parse_vehicle_state


class WaymoMapManager(BaseManager):
    PRIORITY = 0  # Map update has the most high priority
    DEFAULT_DATA_BUFFER_SIZE = 200

    def __init__(self):
        super(WaymoMapManager, self).__init__()
        self.store_map = self.engine.global_config.get("store_map", False)
        # store_map_buffer_size = self.engine.global_config.get("store_map_buffer_size", self.DEFAULT_DATA_BUFFER_SIZE)
        self.current_map = None
        self.map_num = self.engine.global_config["scenario_num"]
        self.start_scenario_index = self.engine.global_config["start_scenario_index"]
        self._stored_maps = {i: None for i in range(self.start_scenario_index, self.start_scenario_index + self.map_num)}

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

        if self._stored_maps[seed] is None and self.store_map:
            new_map = WaymoMap(map_index=seed)
            self._stored_maps[seed] = new_map
        else:
            new_map = self._stored_maps[seed]
        self.load_map(new_map)

        self.update_route()

    def update_route(self):
        data = self.engine.data_manager.get_scenario(self.engine.global_random_seed)

        sdc_track = data.get_sdc_track()

        sdc_traj = parse_full_trajectory(
            sdc_track, coordinate_transform=self.engine.global_config["coordinate_transform"]
        )

        init_state = parse_vehicle_state(
            sdc_track,
            self.engine.global_config["traj_start_index"],
            check_last_state=False,
            coordinate_transform=self.engine.global_config["coordinate_transform"]
        )
        last_state = parse_vehicle_state(
            sdc_track,
            self.engine.global_config["traj_end_index"],
            check_last_state=True,
            coordinate_transform=self.engine.global_config["coordinate_transform"]
        )
        init_position = init_state["position"]
        init_yaw = init_state["heading"]
        last_position = last_state["position"]
        last_yaw = last_state["heading"]

        self.current_sdc_route = copy.copy(PointLane(sdc_traj, 1.5))

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
        raise ValueError("Please create WaymoMap instance directly without calling spawn_object function.")
        # map = self.engine.spawn_object(object_class, auto_fill_random_seed=False, *args, **kwargs)
        # self.spawned_objects[map.id] = map
        # return map

    def load_map(self, map):
        map.attach_to_world()
        self.current_map = map

    def unload_map(self, map):
        map.detach_from_world()
        self.current_map = None
        # if not self.engine.global_config["store_map"]:
        #     self.clear_objects([map.id])
        #     assert len(self.spawned_objects) == 0

    def destroy(self):
        self.maps = None
        self.current_map = None

        self.sdc_start = None
        self.sdc_destinations = []
        self.sdc_dest_point = None
        self.current_sdc_route = None

        super(WaymoMapManager, self).destroy()

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
        return super(WaymoMapManager, self).clear_objects(force_destroy=True, *args, **kwargs)
