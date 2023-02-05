from metadrive.component.map.nuplan_map import NuPlanMap
from metadrive.constants import DEFAULT_AGENT
from metadrive.manager.base_manager import BaseManager
from metadrive.utils.data_buffer import DataBuffer


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

        self.store_map_buffer = DataBuffer(store_data_buffer_size=store_map_buffer_size if self.store_map else None)

    def reset(self):
        seed = self.engine.global_random_seed
        # TODO(LQY) add assert
        # assert self.start <= seed < self.start + self.map_num

        if seed in self.store_map_buffer:
            new_map = self.store_map_buffer[seed]
        else:
            self.store_map_buffer.clear_if_necessary()
            new_map = NuPlanMap(map_index=0)
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

    def before_reset(self):
        # remove map from world before adding
        if self.current_map is not None:
            self.unload_map(self.current_map)

    def clear_objects(self, *args, **kwargs):
        """
        As Map instance should not be recycled, we will forcefully destroy useless map instances.
        """
        return super(NuPlanMapManager, self).clear_objects(force_destroy=True, *args, **kwargs)

    def update_ego_route(self):
        """
        # TODO(LQY) Modify this part, if we finally deceide to use TrajNavi
        """
        data = self.engine.data_manager.get_case(self.engine.global_random_seed)

        ########################TODO(LQY) Fix this after building traffic manager
        # sdc_traj = WaymoTrafficManager.parse_full_trajectory(data["tracks"][data["sdc_index"]]["state"])
        # self.current_sdc_route = PointLane(sdc_traj, 1.5)
        # init_state = WaymoTrafficManager.parse_vehicle_state(
        #     data["tracks"][data["sdc_index"]]["state"],
        #     self.engine.global_config["traj_start_index"],
        #     check_last_state=False,
        # )
        # last_state = WaymoTrafficManager.parse_vehicle_state(
        #     data["tracks"][data["sdc_index"]]["state"],
        #     self.engine.global_config["traj_end_index"],
        #     check_last_state=True
        # )

        # init_position = init_state["position"]
        # init_yaw = init_state["heading"]
        # last_position = last_state["position"]
        # last_yaw = last_state["heading"]
        # self.sdc_dest_point = last_position
        ##############################

        # TODO(LQY):
        #  The Following part is for EdgeNetworkNavi
        #  Consider removing them if we finally choose to use TrajectoryNavi
        # start_lanes = ray_localization(
        #     [np.cos(init_yaw), np.sin(init_yaw)],
        #     init_position,
        #     self.engine,
        #     return_all_result=True,
        #     use_heading_filter=False
        # )
        # end_lanes = ray_localization(
        #     [np.cos(last_yaw), np.sin(last_yaw)],
        #     last_position,
        #     self.engine,
        #     return_all_result=True,
        #     use_heading_filter=False
        # )
        #
        # self.sdc_start, sdc_end = self.filter_path(start_lanes, end_lanes)
        # initial_long, initial_lat = self.current_map.road_network. \
        #     get_lane(self.sdc_start).local_coordinates(init_position)



        # self.sdc_destinations = [sdc_end]
        # lane = self.current_map.road_network.get_lane(sdc_end)
        # if len(lane.left_lanes) > 0:
        #     self.sdc_destinations += [lane["id"] for lane in lane.left_lanes]
        # if len(lane.right_lanes) > 0:
        #     self.sdc_destinations += [lane["id"] for lane in lane.right_lanes]
        start = [664396.54429387, 3997613.41534655]
        end = [ 664396.30707505, 3997613.49425936]
        self.engine.global_config.update(
            dict(target_vehicle_configs={DEFAULT_AGENT: dict(spawn_position_heading=(start, 0))})
        )
