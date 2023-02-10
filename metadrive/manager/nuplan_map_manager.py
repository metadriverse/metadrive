from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario

from metadrive.component.lane.point_lane import PointLane
from metadrive.component.map.nuplan_map import NuPlanMap
from metadrive.constants import DEFAULT_AGENT
from metadrive.manager.base_manager import BaseManager
from metadrive.utils.data_buffer import DataBuffer
from metadrive.utils.nuplan_utils.parse_traffic import parse_ego_vehicle_trajectory, parse_ego_vehicle_state


class NuPlanMapManager(BaseManager):
    """
    Build to execute the same function as NuPlanMapManager
    """
    PRIORITY = 0  # Map update has the most high priority
    DEFAULT_DATA_BUFFER_SIZE = 200

    def __init__(self):
        super(NuPlanMapManager, self).__init__()
        self.store_map = self.engine.global_config.get("store_map", False)
        store_map_buffer_size = self.engine.global_config.get("store_map_buffer_size", self.DEFAULT_DATA_BUFFER_SIZE)
        self.current_map = None
        self.map_num = self.engine.data_manager.scenario_num
        self.start = self.engine.global_config["start_case_index"]
        self.sdc_dest_point = None
        self.current_sdc_route = None

        self.store_map_buffer = DataBuffer(store_data_buffer_size=store_map_buffer_size if self.store_map else None)

    def reset(self):
        seed = self.engine.global_random_seed
        assert self.start <= seed < self.start + self.map_num

        if seed in self.store_map_buffer:
            new_map = self.store_map_buffer[seed]
        else:
            self.store_map_buffer.clear_if_necessary()
            state = self.engine.data_manager.current_scenario.get_ego_state_at_iteration(0)
            center = [state.waypoint.x, state.waypoint.y]
            new_map = NuPlanMap(nuplan_center=center, map_index=seed)
            self.store_map_buffer[seed] = new_map

        self.load_map(new_map)
        self.update_ego_route()

    def filter_path(self, start_lanes, end_lanes):
        for start in start_lanes:
            for end in end_lanes:
                path = self.current_map.road_network.shortest_path(start[0].index, end[0].index)
                if len(path) > 0:
                    return (start[0].index, end[0].index)
        return None

    def spawn_object(self, object_class, *args, **kwargs):
        raise ValueError("Please create NuPlanMap instance directly without calling spawn_object function.")
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
        self.current_map = None
        super(NuPlanMapManager, self).destroy()
        self.sdc_dest_point = None
        self.current_sdc_route = None

    def before_reset(self):
        # remove map from world before adding
        if self.current_map is not None:
            self.unload_map(self.current_map)
        self.sdc_dest_point = None
        self.current_sdc_route = None

    def clear_objects(self, *args, **kwargs):
        """
        As Map instance should not be recycled, we will forcefully destroy useless map instances.
        """
        return super(NuPlanMapManager, self).clear_objects(force_destroy=True, *args, **kwargs)

    def update_ego_route(self):
        """
        Ego Route is placed in map manager
        """
        scenario: NuPlanScenario = self.engine.data_manager.get_case(self.engine.global_random_seed)

        sdc_traj = parse_ego_vehicle_trajectory(scenario.get_expert_ego_trajectory(), self.current_map.nuplan_center)
        self.current_sdc_route = PointLane(sdc_traj, 1.5)
        init_state = parse_ego_vehicle_state(scenario.get_ego_state_at_iteration(0), self.current_map.nuplan_center)
        last_state = parse_ego_vehicle_state(
            scenario.get_ego_state_at_iteration(scenario.get_number_of_iterations() - 1), self.current_map.nuplan_center
        )

        init_position = init_state["position"]
        init_yaw = init_state["heading"]
        last_position = last_state["position"]

        self.sdc_dest_point = last_position
        self.engine.global_config.update(
            dict(
                target_vehicle_configs={
                    DEFAULT_AGENT: dict(
                        spawn_position_heading=(init_position, init_yaw),
                        spawn_velocity=init_state["velocity"],
                        spawn_velocity_car_frame=True,
                    )
                }
            )
        )
