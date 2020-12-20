import logging
from collections import deque, namedtuple
from typing import List, Tuple

import numpy as np
import pandas as pd
from panda3d.bullet import BulletWorld
from pgdrive.scene_creator.map import Map
from pgdrive.scene_creator.road_object.object import RoadObject
from pgdrive.utils import norm, get_np_random
from pgdrive.world.pg_world import PgWorld

logger = logging.getLogger(__name__)

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]

BlockVehicles = namedtuple("block_vehicles", "trigger_road vehicles")


class TrafficMode:
    Reborn = 0
    Add_once = 1


class TrafficManager:
    """Manage all traffic vehicles"""
    VEHICLE_GAP = 10  # m

    def __init__(self, traffic_mode=TrafficMode.Add_once, random_traffic: bool = False):
        """
        :param traffic_mode: reborn/trigger mode
        :param random_traffic: if True, map seed is different with traffic manager seed
        """
        self.traffic_mode = traffic_mode
        self.random_traffic = random_traffic
        self.block_triggered_vehicles = [] if self.traffic_mode == TrafficMode.Add_once else None
        self.blocks = None
        self.network = None
        self.reborn_lanes = None
        self.traffic_density = None
        self.vehicles = None
        self.ego_vehicle = None
        self.traffic_vehicles = None
        self.objects = None
        self.np_random = None
        self.random_seed = None

    def generate_traffic(
        self, pg_world: PgWorld, map: Map, ego_vehicle, traffic_density: float, road_objects: List = None
    ):
        """
        For garbage collecting using, ensure to release the memory of all traffic vehicles
        """
        random_seed = map.random_seed if not self.random_traffic else None
        logging.debug("load scene {}".format(random_seed))
        self.clear_traffic(pg_world.physics_world)
        self.ego_vehicle = ego_vehicle
        self.block_triggered_vehicles = [] if self.traffic_mode == TrafficMode.Add_once else None
        self.blocks = map.blocks
        self.network = map.road_network
        self.reborn_lanes = self._get_available_reborn_lanes()
        self.traffic_density = traffic_density
        self.vehicles = [ego_vehicle]  # it is used to perform IDM and bicycle model based motion
        self.traffic_vehicles = deque()  # it is used to step all vehicles on scene
        self.objects = road_objects or []
        self.random_seed = random_seed
        self.np_random = get_np_random(self.random_seed)
        self.add_vehicles(pg_world)

    def clear_traffic(self, pg_physics_world: BulletWorld):
        if self.traffic_vehicles is not None:
            for v in self.traffic_vehicles:
                v.destroy(pg_physics_world)
        if self.block_triggered_vehicles is not None:
            for block_vs in self.block_triggered_vehicles:
                for v in block_vs.vehicles:
                    v.destroy(pg_physics_world)

    def add_vehicles(self, pg_world):
        if abs(self.traffic_density - 0.0) < 1e-2:
            return
        if self.traffic_mode == TrafficMode.Reborn:
            # add reborn vehicle
            for lane in self.reborn_lanes:
                self.traffic_vehicles += self._create_vehicles_on_lane(lane, True)
            for vehicle in self.traffic_vehicles:
                vehicle.attach_to_pg_world(pg_world.pbr_worldNP, pg_world.physics_world)
            logging.debug("Init {} Traffic Vehicles".format(len(self.traffic_vehicles)))
        else:
            self._create_vehicles_once(pg_world)

    def _create_vehicles_on_lane(self, lane, is_reborn_lane=False):
        from pgdrive.scene_creator.pg_traffic_vehicle.traffic_vehicle_type import car_type
        from pgdrive.scene_creator.blocks.ramp import InRampOnStraight
        traffic_vehicles = []
        total_num = int(lane.length / self.VEHICLE_GAP)
        vehicle_longs = [i * self.VEHICLE_GAP for i in range(total_num)]
        self.np_random.shuffle(vehicle_longs)
        for long in vehicle_longs:
            if self.np_random.rand() > self.traffic_density and abs(lane.length - InRampOnStraight.RAMP_LEN) > 0.1:
                # Do special handling for ramp, and there must be vehicles created there
                continue
            vehicle_type = car_type[self.np_random.choice(list(car_type.keys()), p=[0.5, 0.3, 0.2])]
            random_v = vehicle_type.create_random_traffic_vehicle(
                self, lane, long, seed=self.random_seed, enable_reborn=is_reborn_lane
            )
            self.vehicles.append(random_v.vehicle_node.kinematic_model)
            traffic_vehicles.append(random_v)
        return traffic_vehicles

    def _create_vehicles_once(self, pg_world):
        vehicle_num = 0
        for block in self.blocks[1:]:
            vehicles_on_block = []
            trigger_road = block._pre_block_socket.positive_road

            # trigger lanes is a two dimension array [[]], the first dim represent road consisting of lanes.
            trigger_lanes = block.block_network.get_positive_lanes()
            reborn_lanes = block.get_reborn_lanes()
            for lanes in reborn_lanes:
                if lanes not in trigger_lanes:
                    trigger_lanes.append(lanes)
            self.np_random.shuffle(trigger_lanes)
            for lanes in trigger_lanes:
                num = min(int(len(lanes) * self.traffic_density) + 1, len(lanes))
                lanes = self.np_random.choice(lanes, num, replace=False) if len(lanes) != 1 else lanes
                for l in lanes:
                    vehicles_on_block += self._create_vehicles_on_lane(l)
            for vehicle in vehicles_on_block:
                vehicle.attach_to_pg_world(pg_world.pbr_worldNP, pg_world.physics_world)
            block_vehicles = BlockVehicles(trigger_road=trigger_road, vehicles=vehicles_on_block)
            self.block_triggered_vehicles.append(block_vehicles)
            vehicle_num += len(vehicles_on_block)
        logging.debug("Init {} Traffic Vehicles".format(vehicle_num))
        self.block_triggered_vehicles.reverse()

    def _get_available_reborn_lanes(self):
        reborn_lanes = []
        reborn_roads = []
        for block in self.blocks:
            roads = block.get_reborn_roads()
            for road in roads:
                if road in reborn_roads:
                    reborn_roads.remove(road)
                else:
                    reborn_roads.append(road)
        for road in reborn_roads:
            reborn_lanes += road.get_lanes(self.network)
        return reborn_lanes

    def close_vehicles_to(self, vehicle, distance: float, count: int = None, see_behind: bool = True) -> object:
        vehicles = [
            v for v in self.vehicles
            if norm((v.position - vehicle.position)[0], (v.position - vehicle.position)[1]) < distance
            and v is not vehicle and (see_behind or -2 * vehicle.LENGTH < vehicle.lane_distance_to(v))
        ]

        vehicles = sorted(vehicles, key=lambda v: abs(vehicle.lane_distance_to(v)))
        if count:
            vehicles = vehicles[:count]
        return vehicles

    def prepare_step(self):
        from pgdrive.scene_creator.road.road import Road
        if self.traffic_mode == TrafficMode.Add_once:
            ego_lane_idx = self.ego_vehicle.lane_index[:-1]
            ego_road = Road(ego_lane_idx[0], ego_lane_idx[1])
            if len(self.block_triggered_vehicles) > 0 and ego_road == self.block_triggered_vehicles[-1].trigger_road:
                block_vehicles = self.block_triggered_vehicles.pop()
                self.traffic_vehicles += block_vehicles.vehicles
        for v in self.traffic_vehicles:
            v.prepare_step()

    def step(self, dt: float) -> None:
        """
        Step the dynamics of each entity on the road.

        :param dt: timestep [s]
        """
        dt /= 3.6  # 1m/s = 3.6km/h
        for v in self.traffic_vehicles:
            v.step(dt)

    def update_state(self, pg_physics_world: BulletWorld):
        vehicles_to_remove = []
        for v in self.traffic_vehicles:
            if v.out_of_road():
                remove = v.need_remove()
                if remove:
                    vehicles_to_remove.append(v)
            else:
                v.update_state()

        # remove vehicles out of road
        for v in vehicles_to_remove:
            self.traffic_vehicles.remove(v)
            self.vehicles.remove(v.vehicle_node.kinematic_model)
            v.destroy(pg_physics_world)

    def neighbour_vehicles(self, vehicle, lane_index: LaneIndex = None) -> Tuple:
        """
        Find the preceding and following vehicles of a given vehicle.

        :param vehicle: the vehicle whose neighbours must be found
        :param lane_index: the lane on which to look for preceding and following vehicles.
                     It doesn't have to be the current vehicle lane but can also be another lane, in which case the
                     vehicle is projected on it considering its local coordinates in the lane.
        :return: its preceding vehicle, its following vehicle
        """
        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None
        lane = self.network.get_lane(lane_index)
        s = self.network.get_lane(lane_index).local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        for v in self.vehicles + self.objects:
            if v is not vehicle and not isinstance(v, RoadObject):  # self.network.is_connected_road(v.lane_index,
                # lane_index, same_lane=True):
                s_v, lat_v = lane.local_coordinates(v.position)
                if not lane.on_lane(v.position, s_v, lat_v, margin=1):
                    continue
                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v
        return v_front, v_rear

    def dump(self) -> None:
        """Dump the data of all entities on the road."""
        for v in self.vehicles:
            v.dump()

    def get_log(self) -> pd.DataFrame:
        """
        Concatenate the logs of all entities on the road.

        :return: the concatenated log.
        """
        return pd.concat([v.get_log() for v in self.vehicles])

    def destroy(self, pg_physics_world: BulletWorld):
        self.clear_traffic(pg_physics_world)
        self.blocks = None
        self.network = None
        self.reborn_lanes = None
        self.traffic_density = None
        self.vehicles = None
        self.traffic_vehicles = None
        self.objects = None
        self.np_random = None
        self.random_seed = None

    def __repr__(self):
        return self.vehicles.__repr__()

    def __del__(self):
        logging.debug("{} is destroyed".format(self.__class__.__name__))

    def get_vehicle_num(self):
        if self.traffic_mode == TrafficMode.Reborn:
            return len(self.traffic_vehicles)
        return sum(len(block_vehicle_set.vehicles) for block_vehicle_set in self.block_triggered_vehicles)
