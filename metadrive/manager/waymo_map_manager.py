from metadrive.component.lane.point_lane import PointLane
from metadrive.component.map.waymo_map import WaymoMap
from metadrive.constants import DEFAULT_AGENT
from metadrive.manager.base_manager import BaseManager
from metadrive.manager.waymo_traffic_manager import WaymoTrafficManager
from metadrive.utils.data_buffer import DataBuffer


class WaymoMapManager(BaseManager):
    PRIORITY = 0  # Map update has the most high priority
    DEFAULT_DATA_BUFFER_SIZE = 200

    def __init__(self):
        super(WaymoMapManager, self).__init__()
        store_map = self.engine.global_config.get("store_map", False)
        store_map_buffer_size = self.engine.global_config.get("store_map_buffer_size", self.DEFAULT_DATA_BUFFER_SIZE)
        self.current_map = None
        self.map_num = self.engine.global_config["case_num"]
        self.start = self.engine.global_config["start_case_index"]

        # we put the route searching function here
        self.sdc_start = None
        self.sdc_destinations = []
        self.sdc_dest_point = None
        self.current_sdc_route = None

        self.store_map = store_map
        self.store_map_buffer = DataBuffer(store_data_buffer_size=store_map_buffer_size if self.store_map else None)

    def reset(self):
        seed = self.engine.global_random_seed
        assert self.start <= seed < self.start + self.map_num

        # inner psutil function
        # def process_memory():
        #     import psutil
        #     import os
        #     process = psutil.Process(os.getpid())
        #     mem_info = process.memory_info()
        #     return mem_info.rss
        #
        # cm = process_memory()

        if seed in self.store_map_buffer:
            new_map = self.store_map_buffer[seed]
        else:

            # lm = process_memory()
            # print("{}:  Reset! Mem Change {:.3f}MB".format(1, (lm - cm) / 1e6))
            # cm = lm

            self.store_map_buffer.clear_if_necessary()

            # lm = process_memory()
            # print("{}:  Reset! Mem Change {:.3f}MB".format(2, (lm - cm) / 1e6))
            # cm = lm

            new_map = WaymoMap(map_index=seed)

            # lm = process_memory()
            # print("{}:  Reset! Mem Change {:.3f}MB".format(3, (lm - cm) / 1e6))
            # cm = lm

            self.store_map_buffer[seed] = new_map

            # lm = process_memory()
            # print("{}:  Reset! Mem Change {:.3f}MB".format(4, (lm - cm) / 1e6))
            # cm = lm

        self.load_map(new_map)

        # lm = process_memory()
        # print("{}:  Reset! Mem Change {:.3f}MB".format(5, (lm - cm) / 1e6))
        # cm = lm

        self.update_route()

        # lm = process_memory()
        # print("{}:  Reset! Mem Change {:.3f}MB".format(6, (lm - cm) / 1e6))
        # cm = lm

    def update_route(self):
        """
        # TODO(LQY) Modify this part, if we finally deceide to use TrajNavi
        """
        data = self.engine.data_manager.get_case(self.engine.global_random_seed)

        sdc_traj = WaymoTrafficManager.parse_full_trajectory(data["tracks"][data["sdc_index"]]["state"])
        self.current_sdc_route = PointLane(sdc_traj, 1.5)
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
            dict(
                target_vehicle_configs={
                    DEFAULT_AGENT: dict(
                        spawn_position_heading=(init_position, init_yaw), spawn_velocity=init_state["velocity"]
                    )
                }
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

    def clear_objects(self, *args, **kwargs):
        """
        As Map instance should not be recycled, we will forcefully destroy useless map instances.
        """
        return super(WaymoMapManager, self).clear_objects(force_destroy=True, *args, **kwargs)
