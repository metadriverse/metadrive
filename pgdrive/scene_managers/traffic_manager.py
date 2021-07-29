import copy
import logging
from collections import namedtuple, deque
from typing import Tuple, Dict

from pgdrive.constants import TARGET_VEHICLES, TRAFFIC_VEHICLES, OBJECT_TO_AGENT, AGENT_TO_OBJECT
from pgdrive.scene_creator.lane.abs_lane import AbstractLane
from pgdrive.scene_creator.map.map import Map
from pgdrive.scene_creator.road.road import Road
from pgdrive.scene_managers.base_manager import BaseManager
from pgdrive.utils import norm, merge_dicts
from pgdrive.utils.engine_utils import get_pgdrive_engine

BlockVehicles = namedtuple("block_vehicles", "trigger_road vehicles")


class TrafficMode:
    # Traffic vehicles will be respawn, once they arrive at the destinations
    Respawn = "respawn"

    # Traffic vehicles will be triggered only once
    Trigger = "trigger"

    # Hybrid, some vehicles are triggered once on map and disappear when arriving at destination, others exist all time
    Hybrid = "hybrid"


class TrafficManager(BaseManager):
    VEHICLE_GAP = 10  # m

    def __init__(self):
        """
        Control the whole traffic flow
        """
        super(TrafficManager, self).__init__()
        self.current_map = None

        self._traffic_vehicles = None
        self.block_triggered_vehicles = None
        self.is_target_vehicle_dict = {}

        # traffic property
        self.mode = self.pgdrive_engine.global_config["traffic_mode"]
        self.random_traffic = self.pgdrive_engine.global_config["random_traffic"]
        self.density = self.pgdrive_engine.global_config["traffic_density"]
        self.respawn_lanes = None

    def reset(self):
        """
        Generate traffic on map, according to the mode and density
        :return: List of Traffic vehicles
        """
        map = self.current_map
        logging.debug("load scene {}, {}".format(map.random_seed, "Use random traffic" if self.random_traffic else ""))

        # update vehicle list
        self.block_triggered_vehicles = [] if self.mode != TrafficMode.Respawn else None
        for v in self.vehicles:
            self.is_target_vehicle_dict[v.name] = True

        traffic_density = self.density
        if abs(traffic_density - 0.0) < 1e-2:
            return
        self.respawn_lanes = None
        if self.mode == TrafficMode.Respawn:
            # add respawn vehicle
            self._create_respawn_vehicles(map, traffic_density)
        elif self.mode == TrafficMode.Trigger:
            self._create_vehicles_once(map, traffic_density)
        elif self.mode == TrafficMode.Hybrid:
            # vehicles will be respawn after arriving destination
            self.respawn_lanes = self._get_available_respawn_lanes(map)
            self._create_vehicles_once(map, traffic_density)
        else:
            raise ValueError("No such mode named {}".format(self.mode))
        logging.debug("Init {} Traffic Vehicles".format(len(self._spawned_objects)))

    def before_step(self):
        """
        All traffic vehicles make driving decision here
        :return: None
        """
        # trigger vehicles
        engine = self.pgdrive_engine
        if self.mode != TrafficMode.Respawn:
            for v in engine.agent_manager.active_objects.values():
                ego_lane_idx = v.lane_index[:-1]
                ego_road = Road(ego_lane_idx[0], ego_lane_idx[1])
                if len(self.block_triggered_vehicles) > 0 and \
                        ego_road == self.block_triggered_vehicles[-1].trigger_road:
                    block_vehicles = self.block_triggered_vehicles.pop()
                    self._traffic_vehicles += block_vehicles.vehicles
        for v in self._traffic_vehicles:
            p = self.pgdrive_engine.policy_manager.get_policy(v.name)
            # TODO(pzh): Why we input None here? Is that correct?
            p.before_step(vehicle=v, front_vehicle=None, rear_vehicle=None, current_map=engine.current_map)
            v.before_step()

    def step(self):
        """
        Move all traffic vehicles
        :param dt: Decision keeping time
        :return: None
        """
        dt = self.pgdrive_engine.world_config["physics_world_step_size"]
        dt /= 3.6  # 1m/s = 3.6km/h
        for v in self._traffic_vehicles:
            p = self.pgdrive_engine.policy_manager.get_policy(v.name)
            action = p.step(dt)
            v.step(dt, action)

    def after_step(self):
        """
        Update all traffic vehicles' states,
        """
        vehicles_to_remove = []
        for v in self._traffic_vehicles:
            p = self.pgdrive_engine.policy_manager.get_policy(v.name)
            p.after_step()
            if v.out_of_road:
                remove = v.need_remove()
                if remove:
                    vehicles_to_remove.append(v)
                else:
                    v.reset()
            else:
                v.after_step()

        # remove vehicles out of road
        for v in vehicles_to_remove:
            self._traffic_vehicles.remove(v)
            v.destroy()
            self._spawned_objects.pop(v.id)

            if self.mode == TrafficMode.Hybrid:
                # create a new one
                lane = self.np_random.choice(self.respawn_lanes)
                vehicle_type = self.random_vehicle_type()
                self.spawn_object(vehicle_type, lane, self.np_random.rand() * lane.length / 2, True)

    def clear_objects(self, filter_func=None):
        super(TrafficManager, self).clear_objects()
        self._traffic_vehicles = deque()  # it is used to step all vehicles on scene

    def before_reset(self) -> None:
        """
        Clear the scene and then reset the scene to empty
        :return: None
        """
        # update global info
        self.current_map = self.pgdrive_engine.map_manager.current_map
        self.density = self.pgdrive_engine.global_config["traffic_density"]
        self.clear_objects()

        self.is_target_vehicle_dict.clear()
        self.block_triggered_vehicles = [] if self.mode != TrafficMode.Respawn else None
        self._traffic_vehicles = deque()  # it is used to step all vehicles on scene

    def get_vehicle_num(self):
        """
        Get the vehicles on road
        :return:
        """
        if self.mode == TrafficMode.Respawn:
            return len(self._traffic_vehicles)
        return sum(len(block_vehicle_set.vehicles) for block_vehicle_set in self.block_triggered_vehicles)

    def get_global_states(self) -> Dict:
        """
        Return all traffic vehicles' states
        :return: States of all vehicles
        """
        states = dict()
        traffic_states = dict()
        for vehicle in self._traffic_vehicles:
            traffic_states[vehicle.index] = vehicle.get_state()

        # collect other vehicles
        if self.mode != TrafficMode.Respawn:
            for v_b in self.block_triggered_vehicles:
                for vehicle in v_b.vehicles:
                    traffic_states[vehicle.index] = vehicle.get_state()
        states[TRAFFIC_VEHICLES] = traffic_states
        active_obj = copy.copy(self.pgdrive_engine.agent_manager._active_objects)
        pending_obj = copy.copy(self.pgdrive_engine.agent_manager._pending_objects)
        dying_obj = copy.copy(self.pgdrive_engine.agent_manager._dying_objects)
        states[TARGET_VEHICLES] = {k: v.get_state() for k, v in active_obj.items()}
        states[TARGET_VEHICLES] = merge_dicts(
            states[TARGET_VEHICLES], {k: v.get_state()
                                      for k, v in pending_obj.items()}, allow_new_keys=True
        )
        states[TARGET_VEHICLES] = merge_dicts(
            states[TARGET_VEHICLES], {k: v_count[0].get_state()
                                      for k, v_count in dying_obj.items()},
            allow_new_keys=True
        )

        states[OBJECT_TO_AGENT] = copy.deepcopy(self.pgdrive_engine.agent_manager._object_to_agent)
        states[AGENT_TO_OBJECT] = copy.deepcopy(self.pgdrive_engine.agent_manager._agent_to_object)
        return states

    def get_global_init_states(self) -> Dict:
        """
        Special handling for first states of traffic vehicles
        :return: States of all vehicles
        """
        vehicles = dict()
        for vehicle in self._traffic_vehicles:
            init_state = vehicle.get_state()
            init_state["index"] = vehicle.index
            init_state["type"] = vehicle.class_name
            init_state["enable_respawn"] = vehicle.enable_respawn
            vehicles[vehicle.index] = init_state

        # collect other vehicles
        if self.mode != TrafficMode.Respawn:
            for v_b in self.block_triggered_vehicles:
                for vehicle in v_b.vehicles:
                    init_state = vehicle.get_state()
                    init_state["type"] = vehicle.class_name
                    init_state["index"] = vehicle.index
                    init_state["enable_respawn"] = vehicle.enable_respawn
                    vehicles[vehicle.index] = init_state
        return vehicles

    def spawn_object(self, vehicle_type, lane: AbstractLane, long: float, enable_respawn: bool, *args, **kwargs):
        """
        Create one vehicle on lane and a specific place
        :param vehicle_type: PGTrafficVehicle type (s,m,l,xl)
        :param lane: Straight Lane or Circular Lane
        :param long: longitude position on lane
        :param enable_respawn: Respawn or not
        :return: PGTrafficVehicle
        """
        random_v = vehicle_type.create_random_traffic_vehicle(
            len(self._spawned_objects), self, lane, long, random_seed=self.randint(), enable_respawn=enable_respawn
        )
        self._spawned_objects[random_v.id] = random_v
        self._traffic_vehicles.append(random_v)

        # TODO(pzh): Clean this part!
        # TODO(pzh): Check whether delay_time is correct!
        # TODO(pzh): Check whether the random seed is correct!
        # Register the IDM policy for each traffic vehicle
        from pgdrive.policy.idm_policy import IDMPolicy
        e = get_pgdrive_engine()
        e.policy_manager.register_new_policy(
            IDMPolicy, vehicle=random_v, traffic_manager=self, random_seed=0, delay_time=1, target_speed=random_v.speed
        )

        return random_v

    def _create_vehicles_on_lane(self, traffic_density: float, lane: AbstractLane, is_respawn_lane):
        """
        Create vehicles on a lane
        :param traffic_density: traffic density according to num of vehicles per meter
        :param lane: Circular lane or Straight lane
        :param is_respawn_lane: Whether vehicles should be respawn on this lane or not
        :return: List of vehicles
        """

        from pgdrive.scene_creator.blocks.ramp import InRampOnStraight
        _traffic_vehicles = []
        total_num = int(lane.length / self.VEHICLE_GAP)
        vehicle_longs = [i * self.VEHICLE_GAP for i in range(total_num)]
        self.np_random.shuffle(vehicle_longs)
        for long in vehicle_longs:
            if self.np_random.rand() > traffic_density and abs(lane.length - InRampOnStraight.RAMP_LEN) > 0.1:
                # Do special handling for ramp, and there must be vehicles created there
                continue
            vehicle_type = self.random_vehicle_type()
            self.spawn_object(vehicle_type, lane, long, is_respawn_lane)
        return _traffic_vehicles

    def _create_respawn_vehicles(self, map: Map, traffic_density: float):
        respawn_lanes = self._get_available_respawn_lanes(map)
        engine = get_pgdrive_engine()
        for lane in respawn_lanes:
            self._traffic_vehicles += self._create_vehicles_on_lane(traffic_density, lane, True)
        for vehicle in self._traffic_vehicles:
            vehicle.attach_to_world(engine.pbr_worldNP, engine.physics_world)

    def _create_vehicles_once(self, map: Map, traffic_density: float) -> None:
        """
        Trigger mode, vehicles will be triggered only once, and disappear when arriving destination
        :param map: Map
        :param traffic_density: it can be adjusted each episode
        :return: None
        """
        engine = get_pgdrive_engine()
        vehicle_num = 0
        for block in map.blocks[1:]:
            if block.PROHIBIT_TRAFFIC_GENERATION:
                continue
            vehicles_on_block = []
            trigger_road = block.pre_block_socket.positive_road

            # trigger lanes is a two dimension array [[]], the first dim represent road consisting of lanes.
            trigger_lanes = block.block_network.get_positive_lanes()
            respawn_lanes = block.get_respawn_lanes()
            for lanes in respawn_lanes:
                if lanes not in trigger_lanes:
                    trigger_lanes.append(lanes)
            self.np_random.shuffle(trigger_lanes)
            for lanes in trigger_lanes:
                num = min(int(len(lanes) * traffic_density) + 1, len(lanes))
                lanes = self.np_random.choice(lanes, num, replace=False) if len(lanes) != 1 else lanes
                for l in lanes:
                    vehicles_on_block += self._create_vehicles_on_lane(traffic_density, l, False)
            for vehicle in vehicles_on_block:
                vehicle.attach_to_world(engine.pbr_worldNP, engine.physics_world)
            block_vehicles = BlockVehicles(trigger_road=trigger_road, vehicles=vehicles_on_block)
            self.block_triggered_vehicles.append(block_vehicles)
            vehicle_num += len(vehicles_on_block)
        self.block_triggered_vehicles.reverse()

    def _get_available_respawn_lanes(self, map: Map) -> list:
        """
        Used to find some respawn lanes
        :param map: select spawn lanes from this map
        :return: respawn_lanes
        """
        respawn_lanes = []
        respawn_roads = []
        for block in map.blocks:
            roads = block.get_respawn_roads()
            for road in roads:
                if road in respawn_roads:
                    respawn_roads.remove(road)
                else:
                    respawn_roads.append(road)
        for road in respawn_roads:
            respawn_lanes += road.get_lanes(map.road_network)
        return respawn_lanes

    def close_vehicles_to(self, vehicle, distance: float, count: int = None, see_behind: bool = True) -> object:
        """
        Find the closest vehicles for IDM vehicles
        :param vehicle: IDM vehicle
        :param distance: How much distance
        :param count: Num of vehicles to return
        :param see_behind: Whether find vehicles behind this IDM vehicle or not
        :return:
        """
        raise DeprecationWarning("This func is Deprecated")
        vehicles = [
            v for v in self.vehicles
            if norm((v.position - vehicle.position)[0], (v.position - vehicle.position)[1]) < distance
            and v is not vehicle and (see_behind or -2 * vehicle.LENGTH < vehicle.lane_distance_to(v))
        ]

        vehicles = sorted(vehicles, key=lambda v: abs(vehicle.lane_distance_to(v)))
        if count:
            vehicles = vehicles[:count]
        return vehicles

    def neighbour_vehicles(self, vehicle, lane_index: Tuple = None) -> Tuple:
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
        lane = self.current_map.road_network.get_lane(lane_index)
        s = self.current_map.road_network.get_lane(lane_index).local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        for v in self.vehicles + self.pgdrive_engine.object_manager.objects:
            if norm(v.position[0] - vehicle.position[0], v.position[1] - vehicle.position[1]) > 100:
                # coarse filter
                continue
            if v is not vehicle:
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

    def random_vehicle_type(self):
        from pgdrive.scene_creator.vehicle.traffic_vehicle_type import vehicle_type
        vehicle_type = vehicle_type[self.np_random.choice(list(vehicle_type.keys()), p=[0.2, 0.3, 0.3, 0.2])]
        return vehicle_type

    def destroy(self) -> None:
        """
        Destory func, release resource
        :return: None
        """
        self.clear_objects()
        # current map
        self.current_map = None

        # traffic vehicle list
        self._traffic_vehicles = None
        self.block_triggered_vehicles = None

        # traffic property
        self.mode = None
        self.random_traffic = None
        self.density = None

    def __del__(self):
        logging.debug("{} is destroyed".format(self.__class__.__name__))

    def __repr__(self):
        return self.vehicles.__repr__()

    def is_target_vehicle(self, v):
        if v.name in self.is_target_vehicle_dict and self.is_target_vehicle_dict[v.name]:
            return True
        return False

    @property
    def vehicles(self):
        return list(self.pgdrive_engine.agent_manager.active_objects.values()) + \
               [v.vehicle_node.kinematic_model for v in self._spawned_objects.values()]

    @property
    def traffic_vehicles(self):
        return list(self._traffic_vehicles)

    def seed(self, random_seed):
        if not self.random_traffic:
            super(TrafficManager, self).seed(random_seed)
