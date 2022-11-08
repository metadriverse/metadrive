import logging
from collections import namedtuple

import numpy as np

from metadrive.component.map.base_map import BaseMap
from metadrive.component.vehicle.vehicle_type import *
from metadrive.manager.base_manager import BaseManager

BlockVehicles = namedtuple("block_vehicles", "trigger_road vehicles")


class RealDataManager(BaseManager):
    VEHICLE_GAP = 10  # m

    def __init__(self):
        """
        Replay Argoverse data.
        """
        super(RealDataManager, self).__init__()
        self._traffic_vehicles = []

    def reset(self):
        """
        Generate traffic on map, according to the mode and density
        :return: List of Traffic vehicles
        """
        map = self.current_map
        self._create_argoverse_vehicles_once(map)

    def before_step(self):
        """
        All traffic vehicles make driving decision here
        :return: None
        """
        # trigger vehicles
        engine = self.engine

        for v in self._traffic_vehicles:
            p = self.engine.get_policy(v.name)
            v.before_step(p.act())
        return dict()

    def before_reset(self) -> None:
        """
        Clear the scene and then reset the scene to empty
        :return: None
        """
        super(RealDataManager, self).before_reset()
        self.density = self.engine.global_config["traffic_density"]
        self._traffic_vehicles = []

    def get_vehicle_num(self):
        """
        Get the vehicles on road
        :return:
        """
        return len(self._traffic_vehicles)

    def _create_argoverse_vehicles_once(self, map: BaseMap) -> None:
        """
        Trigger mode, vehicles will be triggered only once, and disappear when arriving destination
        :param map: Map map.road_network[index]
        :param traffic_density: it can be adjusted each episode
        :return: None
        """
        real_data_config = self.engine.global_config["real_data_config"]
        locate_info = real_data_config["locate_info"]
        pos_dict = {i: j["init_pos"] for i, j in zip(locate_info.keys(), locate_info.values())}

        block = map.blocks[0]
        lanes = block.argo_lanes.values()
        roads = self.get_roads(block.block_network, direction='positive', lane_num=1)
        potential_vehicle_configs = []
        for l in lanes:
            start = np.max(l.centerline, axis=0)
            end = np.min(l.centerline, axis=0)
            for idx, pos in zip(pos_dict.keys(), pos_dict.values()):
                v_type = self.random_vehicle_type(prob=[0.4, 0.3, 0.3, 0, 0])
                if start[0] > pos[0] > end[0] and start[1] > pos[1] > end[1]:
                    long, lat = l.local_coordinates(pos)
                    config = {
                        "id": idx,
                        "type": v_type,
                        "v_config": {
                            "spawn_lane_index": l.index,
                            "spawn_longitude": long,
                            "enable_reverse": False,
                        }
                    }
                    potential_vehicle_configs.append(config)
                    pos_dict.pop(idx, None)
                    break
        from metadrive.policy.replay_policy import ReplayPolicy
        for road in roads:
            for config in potential_vehicle_configs:
                v_config = config["v_config"]
                v_start = v_config["spawn_lane_index"][0]
                v_end = v_config["spawn_lane_index"][1]
                v_config.update(self.engine.global_config["traffic_vehicle_config"])
                if road.start_node == v_start and road.end_node == v_end:
                    generated_v = self.spawn_object(config["type"], vehicle_config=v_config)
                    generated_v.set_static(True)
                    self.add_policy(generated_v.id, ReplayPolicy, generated_v, locate_info[config["id"]])
                    self._traffic_vehicles.append(generated_v)
                    potential_vehicle_configs.remove(config)
                    break

    def random_vehicle_type(self, prob=[0.2, 0.3, 0.3, 0.2, 0]):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type
        vehicle_type = random_vehicle_type(self.np_random, prob)
        return vehicle_type

    def destroy(self) -> None:
        """
        Destory func, release resource
        :return: None
        """
        self.clear_objects([v.id for v in self._traffic_vehicles])
        self._traffic_vehicles = []
        # current map

        # traffic vehicle list
        self._traffic_vehicles = None

    def __del__(self):
        logging.debug("{} is destroyed".format(self.__class__.__name__))

    def __repr__(self):
        return self.vehicles.__repr__()

    @property
    def vehicles(self):
        return list(self.engine.get_objects(filter=lambda o: isinstance(o, BaseVehicle)).values())

    @property
    def traffic_vehicles(self):
        return list(self._traffic_vehicles)

    @property
    def current_map(self):
        return self.engine.map_manager.current_map

    @staticmethod
    def get_roads(roadnetwork, *, direction="all", lane_num=None) -> List:
        """
        Return all roads in road_network
        :param direction: "positive"/"negative"
        :param lane_num: only roads with lane_num lanes will be returned
        :return: List[Road]
        """
        assert direction in ["positive", "negative", "all"], "incorrect road direction"
        ret = []
        for _from, _to_dict in self.graph.items():
            if direction == "all" or (direction == "positive" and _from[0] != "-") or (direction == "negative"
                                                                                       and _from[0] == "-"):
                for _to, lanes in _to_dict.items():
                    if lane_num is None or len(lanes) == lane_num:
                        ret.append(Road(_from, _to))
        return ret
