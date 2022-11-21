import numpy as np
from metadrive.component.map.waymo_map import WaymoMap
from metadrive.constants import DEFAULT_AGENT
from metadrive.manager.base_manager import BaseManager
from metadrive.manager.waymo_traffic_manager import WaymoTrafficManager
from metadrive.utils.scene_utils import ray_localization
from metadrive.component.lane.waypoint_lane import WayPointLane


class WaymoMapManager(BaseManager):
    PRIORITY = 0  # Map update has the most high priority

    def __init__(self):
        super(WaymoMapManager, self).__init__()
        self.current_map = None
        self.map_num = self.engine.global_config["case_num"]
        start = self.engine.global_config["start_case_index"]
        self.maps = {_seed: None for _seed in range(start, start + self.map_num)}
        # we put the route-find funcrion here
        self.sdc_start = None
        self.sdc_destinations = []
        self.sdc_dest_point = None
        self.current_sdc_route = None

    def reset(self):
        seed = self.engine.global_random_seed
        map_config = self.engine.data_manager.get_case(seed)
        if self.maps[seed] is None:
            map = self.spawn_object(WaymoMap, waymo_data=map_config)
            if self.engine.global_config["store_map"]:
                self.maps[seed] = map
        else:
            map = self.maps[seed]
        self.load_map(map)
        self.update_route(map_config)

    def update_route(self, data):
        """
        # TODO(LQY) Modify this part, if we finally deceide to use TrajNavi
        """
        sdc_traj = WaymoTrafficManager.parse_full_trajectory(data["tracks"][data["sdc_index"]]["state"])
        self.current_sdc_route = WayPointLane(sdc_traj, 1.5)
        init_state = WaymoTrafficManager.parse_vehicle_state(
            data["tracks"][data["sdc_index"]]["state"],
            self.engine.global_config["traj_start_index"],
            check_last_state=False,
        )
        last_state = WaymoTrafficManager.parse_vehicle_state(
            data["tracks"][data["sdc_index"]]["state"],
            self.engine.global_config["traj_end_index"],
            check_last_state=True
        )
        init_position = init_state["position"]
        init_yaw = init_state["heading"]
        last_position = last_state["position"]
        last_yaw = last_state["heading"]

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

        self.sdc_dest_point = last_position
        # self.sdc_destinations = [sdc_end]
        # lane = self.current_map.road_network.get_lane(sdc_end)
        # if len(lane.left_lanes) > 0:
        #     self.sdc_destinations += [lane["id"] for lane in lane.left_lanes]
        # if len(lane.right_lanes) > 0:
        #     self.sdc_destinations += [lane["id"] for lane in lane.right_lanes]
        self.engine.global_config.update(
            dict(target_vehicle_configs={DEFAULT_AGENT: dict(spawn_position_heading=(init_position, init_yaw))})
        )

    def filter_path(self, start_lanes, end_lanes):
        for start in start_lanes:
            for end in end_lanes:
                path = self.current_map.road_network.shortest_path(start[0].index, end[0].index)
                if len(path) > 0:
                    return (start[0].index, end[0].index)
        return None

    def spawn_object(self, object_class, *args, **kwargs):
        map = self.engine.spawn_object(object_class, auto_fill_random_seed=False, *args, **kwargs)
        self.spawned_objects[map.id] = map
        return map

    def load_map(self, map):
        map.attach_to_world()
        self.current_map = map

    def unload_map(self, map):
        map.detach_from_world()
        self.current_map = None
        if not self.engine.global_config["store_map"]:
            self.clear_objects([map.id], force_destroy=True)
            assert len(self.spawned_objects) == 0

    def destroy(self):
        self.maps = None
        self.current_map = None
        super(WaymoMapManager, self).destroy()

    def before_reset(self):
        # remove map from world before adding
        if self.current_map is not None:
            self.unload_map(self.current_map)
