import logging
from collections import deque, namedtuple
from typing import List, Tuple
from .PgLOD import PgLOD
import pandas as pd

from pgdrive.scene_creator.map import Map
from pgdrive.scene_creator.road.road import Road
from pgdrive.scene_creator.road_object.object import RoadObject
from pgdrive.scene_manager import TrafficMode
from pgdrive.scene_manager.replay_record_system import PGReplayer, PGRecorder
from pgdrive.utils import norm, get_np_random
from pgdrive.world.pg_world import PgWorld

logger = logging.getLogger(__name__)

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]

BlockVehicles = namedtuple("block_vehicles", "trigger_road vehicles")


class SceneManager:
    """Manage all traffic vehicles, and all runtime elements"""
    VEHICLE_GAP = 10  # m

    def __init__(self, traffic_mode=TrafficMode.Trigger, random_traffic: bool = False, record_episode: bool = False):
        """
        :param traffic_mode: reborn/trigger mode
        :param random_traffic: if True, map seed is different with traffic manager seed
        """
        self.record_episode = record_episode
        self.traffic_mode = traffic_mode
        self.random_traffic = random_traffic
        self.block_triggered_vehicles = [] if self.traffic_mode == TrafficMode.Trigger else None
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

        # for recovering, they can not exist together
        self.replay_system = None
        self.record_system = None

    def reset(
        self,
        pg_world: PgWorld,
        map: Map,
        ego_vehicle,
        traffic_density: float,
        road_objects: List = None,
        episode_data=None
    ):
        """
        For garbage collecting using, ensure to release the memory of all traffic vehicles
        """
        random_seed = map.random_seed if not self.random_traffic else None
        logging.debug("load scene {}, {}".format(map.random_seed, "Use random traffic" if self.random_traffic else ""))
        self.clear_traffic(pg_world)
        self.ego_vehicle = ego_vehicle
        self.block_triggered_vehicles = [] if self.traffic_mode == TrafficMode.Trigger else None
        self.blocks = map.blocks
        self.network = map.road_network
        self.reborn_lanes = self._get_available_reborn_lanes()
        self.traffic_density = traffic_density
        self.vehicles = [ego_vehicle]  # it is used to perform IDM and bicycle model based motion
        if self.replay_system is not None:
            self.replay_system.destroy(pg_world)
            self.replay_system = None
        if self.record_system is not None:
            self.record_system.destroy(pg_world)
            self.record_system = None
        self.traffic_vehicles = deque()  # it is used to step all vehicles on scene
        self.objects = road_objects or []
        self.random_seed = random_seed
        self.np_random = get_np_random(self.random_seed)

        if episode_data is None:
            self.add_vehicles(pg_world)
        else:
            self.replay_system = PGReplayer(self, map, episode_data, pg_world)

        if pg_world.highway_render is not None:
            pg_world.highway_render.set_scene_mgr(self)
        if self.record_episode:
            if episode_data is None:
                self.record_system = PGRecorder(map, self.get_global_init_states(), self.traffic_mode)
            else:
                logging.warning("Temporally disable episode recorder, since we are replaying other episode!")

    def clear_traffic(self, pg_world: PgWorld):
        if self.traffic_vehicles is not None:
            for v in self.traffic_vehicles:
                v.destroy(pg_world)
        if self.block_triggered_vehicles is not None:
            for block_vs in self.block_triggered_vehicles:
                for v in block_vs.vehicles:
                    v.destroy(pg_world)

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
        """
        Create vehicles on one lane
        :param lane: Straight lane or Circular lane
        :param is_reborn_lane: Vehicles will be reborn when set to True
        :return: None
        """
        from pgdrive.scene_creator.pg_traffic_vehicle.traffic_vehicle_type import car_type
        from pgdrive.scene_creator.blocks.ramp import InRampOnStraight
        traffic_vehicles = []
        total_num = int(lane.length / self.VEHICLE_GAP)
        vehicle_longs = [i * self.VEHICLE_GAP for i in range(total_num)]
        self.np_random.shuffle(vehicle_longs)
        for i, long in enumerate(vehicle_longs):
            if self.np_random.rand() > self.traffic_density and abs(lane.length - InRampOnStraight.RAMP_LEN) > 0.1:
                # Do special handling for ramp, and there must be vehicles created there
                continue
            vehicle_type = car_type[self.np_random.choice(list(car_type.keys()), p=[0.2, 0.3, 0.3, 0.2])]
            random_v = vehicle_type.create_random_traffic_vehicle(
                len(self.vehicles), self, lane, long, seed=self.random_seed, enable_reborn=is_reborn_lane
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
        if self.traffic_mode == TrafficMode.Trigger:
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

    def update_state(self, pg_world: PgWorld) -> bool:
        # cull distant objects
        PgLOD.cull_distant_blocks(self.blocks, self.ego_vehicle.position, pg_world)
        PgLOD.cull_distant_traffic_vehicles(self.traffic_vehicles, self.ego_vehicle.position, pg_world)

        vehicles_to_remove = []
        for v in self.traffic_vehicles:
            if v.out_of_road:
                remove = v.need_remove()
                if remove:
                    vehicles_to_remove.append(v)
            else:
                v.update_state()

        # remove vehicles out of road
        for v in vehicles_to_remove:
            self.traffic_vehicles.remove(v)
            self.vehicles.remove(v.vehicle_node.kinematic_model)
            v.destroy(pg_world)

        if self.replay_system is not None:
            self.replay_system.replay_frame(self.ego_vehicle, pg_world)
        elif self.record_system is not None:
            # didn't record while replay
            self.record_system.record_frame(self.get_global_states())

        return False

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
            if norm(v.position[0] - vehicle.position[0], v.position[1] - vehicle.position[1]) > 100:
                # coarse filter
                continue
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

    def dump_episode(self) -> None:
        """Dump the data of an episode."""
        assert self.record_system is not None
        return self.record_system.dump_episode()

    def get_log(self) -> pd.DataFrame:
        """
        Concatenate the logs of all entities on the road.

        :return: the concatenated log.
        """
        return pd.concat([v.get_log() for v in self.vehicles])

    def destroy(self, pg_world: PgWorld):
        self.clear_traffic(pg_world)
        self.blocks = None
        self.network = None
        self.reborn_lanes = None
        self.traffic_density = None
        self.vehicles = None
        self.traffic_vehicles = None
        self.objects = None
        self.np_random = None
        self.random_seed = None
        if self.record_system is not None:
            self.record_system.destroy(pg_world)
            self.record_system = None
        if self.replay_system is not None:
            self.replay_system.destroy(pg_world)
            self.replay_system = None

    def __repr__(self):
        return self.vehicles.__repr__()

    def __del__(self):
        logging.debug("{} is destroyed".format(self.__class__.__name__))

    def get_vehicle_num(self):
        if self.traffic_mode == TrafficMode.Reborn:
            return len(self.traffic_vehicles)
        return sum(len(block_vehicle_set.vehicles) for block_vehicle_set in self.block_triggered_vehicles)

    def get_global_states(self):
        states = dict()
        for vehicle in self.traffic_vehicles:
            states[vehicle.index] = vehicle.get_state()

        # collect other vehicles
        if self.traffic_mode == TrafficMode.Trigger:
            for v_b in self.block_triggered_vehicles:
                for vehicle in v_b.vehicles:
                    states[vehicle.index] = vehicle.get_state()

        states["ego"] = self.ego_vehicle.get_state()
        return states

    def get_global_init_states(self):
        vehicles = dict()
        for vehicle in self.traffic_vehicles:
            init_state = vehicle.get_state()
            init_state["index"] = vehicle.index
            init_state["type"] = vehicle.class_name
            vehicles[vehicle.index] = init_state

        # collect other vehicles
        if self.traffic_mode == TrafficMode.Trigger:
            for v_b in self.block_triggered_vehicles:
                for vehicle in v_b.vehicles:
                    init_state = vehicle.get_state()
                    init_state["type"] = vehicle.class_name
                    init_state["index"] = vehicle.index
                    vehicles[vehicle.index] = init_state
        return vehicles
