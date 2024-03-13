import logging
from collections import namedtuple
from typing import Tuple

import numpy as np

from metadrive.manager.traffic_manager import TrafficMode
from metadrive.component.map.base_map import BaseMap
from metadrive.manager.traffic_manager import PGTrafficManager
from metadrive.utils import Vector

BlockVehicles = namedtuple("block_vehicles", "trigger_road vehicles")


from metadrive.policy.adv_policy import AdvPolicy
from copo_scenario.torch_copo.algo_ippo import IPPOTrainer, IPPOPolicy
from metadrive.component.road_network import Road
import random
from metadrive.obs.state_obs import LidarStateObservation


class PGAdversaryVehicleManager(PGTrafficManager):

    def __init__(self):
        """
        Control the whole traffic flow
        """
        super(PGTrafficManager, self).__init__()

        self._traffic_vehicles = []

        # triggered by the event. TODO(lqy) In the future, event trigger can be introduced
        self.block_triggered_vehicles = []

        # traffic property
        self.mode = self.engine.global_config["traffic_mode"]
        self.random_traffic = self.engine.global_config["random_traffic"]
        self.density = self.engine.global_config["traffic_density"]
        self.respawn_lanes = None
        self.num_adversary_vehicles = self.engine.global_config["num_adversary_vehicles"] # TODO: how to pass the config from the environment to the engine config??
        self.observation = self.get_observation()

    def reset(self):
        """
        Generate traffic on map, according to the mode and density
        :return: List of Traffic vehicles
        """
        map = self.current_map
        logging.debug("load scene {}".format("Use random traffic" if self.random_traffic else ""))

        # update vehicle list
        self.block_triggered_vehicles = []

        traffic_density = self.density
        # if abs(traffic_density) < 1e-2:
        #     return
        self.respawn_lanes = self._get_available_respawn_lanes(map)
        if self.mode == TrafficMode.Respawn:
            # add respawn vehicle
            self._create_respawn_vehicles(map, traffic_density)
        elif self.mode == TrafficMode.Trigger or self.mode == TrafficMode.Hybrid:
            self._create_vehicles_once(map, traffic_density)
        elif self.mode == TrafficMode.Adversary: # spawn the adversary vehicles
            # self._create_adversary_vehicles(map, self.num_adversary_vehicles)
            self._create_random_adversary_vehicles(map, self.num_adversary_vehicles)
        else:
            raise ValueError("No such mode named {}".format(self.mode))


    def get_observation(self):
        """
        Override me in the future for collecting other modality
        """
        return LidarStateObservation(self.engine.global_config)


    def before_step(self):
        """
        All traffic vehicles make driving decision here
        :return: None
        """
        # trigger vehicles
        engine = self.engine
        if self.mode != TrafficMode.Respawn:
            for v in engine.agent_manager.active_agents.values():
                ego_lane_idx = v.lane_index[:-1]
                ego_road = Road(ego_lane_idx[0], ego_lane_idx[1])
                # if len(self.block_triggered_vehicles) > 0 and \
                #         ego_road == self.block_triggered_vehicles[-1].trigger_road:
                #     block_vehicles = self.block_triggered_vehicles.pop()
                #     self._traffic_vehicles += list(self.get_objects(block_vehicles.vehicles).values()) # if on the triggered road, then add the vehicles to the traffic vehicles
        for v in self._traffic_vehicles:
            obs = self.observation.observe(vehicle=v)
            v_state = v.get_state()
            p = self.engine.get_policy(v.name)
            if isinstance(p, AdvPolicy):
                action = p.act(obs)
                v.before_step(action)
            else:
                v.before_step(p.act())
        return dict()


    # def _auto_fill_spawn_roads_randomly(self, spawn_roads):
    #     """It is used for shuffling the config"""
    #
    #     num_slots = int(floor(self.exit_length / SpawnManager.RESPAWN_REGION_LONGITUDE))
    #     interval = self.exit_length / num_slots
    #     self._longitude_spawn_interval = interval
    #     if self.num_agents is not None:
    #         assert self.num_agents > 0 or self.num_agents == -1
    #         assert self.num_agents <= self.max_capacity(
    #             spawn_roads, self.exit_length + FirstPGBlock.ENTRANCE_LENGTH, self.lane_num
    #         ), (
    #             "Too many agents! We only accept {} agents, but you have {} agents!".format(
    #                 self.lane_num * len(spawn_roads) * num_slots, self.num_agents
    #             )
    #         )
    #
    #     # We can spawn agents in the middle of road at the initial time, but when some vehicles need to be respawn,
    #     # then we have to set it to the farthest places to ensure safety (otherwise the new vehicles may suddenly
    #     # appear at the middle of the road!)
    #     agent_configs = []
    #     safe_spawn_places = []
    #     for i, road in enumerate(spawn_roads):
    #         for lane_idx in range(self.lane_num):
    #             for j in range(num_slots):
    #                 long = 1 / 2 * self.RESPAWN_REGION_LONGITUDE + j * self.RESPAWN_REGION_LONGITUDE
    #                 lane_tuple = road.lane_index(lane_idx)  # like (>>>, 1C0_0_, 1) and so on.
    #                 agent_configs.append(
    #                     Config(
    #                         dict(
    #                             identifier="|".join((str(s) for s in lane_tuple + (j, ))),
    #                             config={
    #                                 "spawn_lane_index": lane_tuple,
    #                                 "spawn_longitude": long,
    #                                 "spawn_lateral": 0
    #                             },
    #                         ),
    #                         unchangeable=True
    #                     )
    #                 )  # lock the spawn positions
    #                 if j == 0:
    #                     safe_spawn_places.append(copy.deepcopy(agent_configs[-1]))
    #     return agent_configs, safe_spawn_places


    def after_step(self, *args, **kwargs):
        """
        Update all traffic vehicles' states,
        """
        v_to_remove = []
        for v in self._traffic_vehicles:
            v.after_step()
            if not v.on_lane:
                v_to_remove.append(v)
        for v in v_to_remove:
            vehicle_type = type(v)
            self.clear_objects([v.id])
            self._traffic_vehicles.remove(v)
            if self.mode == TrafficMode.Respawn or self.mode == TrafficMode.Hybrid:
                lane = self.respawn_lanes[self.np_random.randint(0, len(self.respawn_lanes))]
                lane_idx = lane.index
                long = self.np_random.rand() * lane.length / 2
                traffic_v_config = {"spawn_lane_index": lane_idx, "spawn_longitude": long}
                new_v = self.spawn_object(vehicle_type, vehicle_config=traffic_v_config)
                from metadrive.policy.idm_policy import IDMPolicy
                self.add_policy(new_v.id, IDMPolicy, new_v, self.generate_seed()) # TODO: should we change the policy type? maybe not
                self._traffic_vehicles.append(new_v)

        return dict()



    def get_adv_config(self, lane, long): # TODO: used for spawn
        return {"spawn_lane_index": lane.index, "spawn_longitude": long, "enable_reverse": False}

    def calculate_long_lat(self, lane, position: Vector) -> Tuple[float, float]:
        # Shift the position by the center of the circle
        shifted_pos_x = position[0] - lane.center[0]
        shifted_pos_y = position[1] - lane.center[1]

        # Calculate phi using atan2
        import math
        phi = math.atan2(shifted_pos_y, shifted_pos_x)

        # Calculate longitudinal
        longitudinal = ((phi - lane.start_phase) * lane.radius) * lane.direction

        # Calculate the distance from the center to the position
        distance_from_center = np.sqrt(shifted_pos_x**2 + shifted_pos_y**2)

        # Calculate lateral, considering the direction
        lateral = (distance_from_center - lane.radius) * lane.direction

        return longitudinal, lateral


    def _create_adversary_vehicles(self, map: BaseMap, num_adversaries: int):
        """
           I also checked agent_manager.py, especially the _create_agents() function
           Trigger mode, vehicles will be triggered only once, and disappear when arriving destination
           :param map: Map
           :param traffic_density: it can be adjusted each episode
           :return: None
           """
        vehicle_num = 0
        first_real_block = map.blocks[1]
        initial_lanes = first_real_block.get_intermediate_spawn_lanes() # this returns two lists of lanes, one for each direction
        spawn_lanes = initial_lanes[0] # this returns all lanes that can be used to spawn objects
        vehicles_on_block = []

        # for lane in spawn_lanes: # TODO: how to choose the lane to spawn the vehicle??
        for i in range(num_adversaries):
            vehicle_type = self.random_vehicle_type() # TODO: adversaries should have their own vehicle type?? Not sure...
            # adv_start_long =  None    #TODO: should we consider about what's the position being trained? Maybe not....
            # adv_config = self.get_adv_config(lane, adv_start_long)
            # adv_start_long = self.np_random.rand() * first_real_block.length
            import random
            spawn_lane = spawn_lanes[random.randint(0, len(spawn_lanes)-1)]
            adv_start_long, adv_start_lat = self.calculate_long_lat(spawn_lane, spawn_lane.start)
            adv_v_config = {"spawn_lane_index": spawn_lane.index, "spawn_longitude": adv_start_long, "spawn_lateral": adv_start_lat, "enable_reverse": False}
            adv_v_config.update(self.engine.global_config["traffic_vehicle_config"])

            # adv_vehicle_config = self.engine.global_config["agent_configs"]["default_agent"] # in the single agent case, this is the same as agent0

            # adv_vehicle_config.update(self.engine.global_config["traffic_vehicle_config"]) # TODO: not necessary??
            random_v = self.spawn_object(vehicle_type, vehicle_config=adv_v_config)
            self._traffic_vehicles.append(random_v)

            # from metadrive.policy.adv_policy import AdvPolicy
            # from metadrive.policy.idm_policy import IDMPolicy
            self.add_policy(random_v.id, AdvPolicy, random_v, self.generate_seed())
            vehicles_on_block.append(random_v.name)



    def _create_random_adversary_vehicles(self, map: BaseMap, num_adversaries: int):
        """
           I also checked agent_manager.py, especially the _create_agents() function
           Trigger mode, vehicles will be triggered only once, and disappear when arriving destination
           :param map: Map
           :param traffic_density: it can be adjusted each episode
           :return: None
           """
        total_vehicles = num_adversaries
        vehicle_num = 0
        vehicles_on_block = []
        # use_neg = False
        for _ in range(num_adversaries):
            use_neg = False
        # for block in map.blocks[1:]:
            block_i = random.randint(1, len(map.blocks)-1)
            block = map.blocks[block_i]
            # Propose candidate locations for spawning new vehicles
            trigger_lanes = block.get_intermediate_spawn_lanes()
            if self.engine.global_config["need_inverse_traffic"] and block.ID in ["S", "C", "r", "R"]:
                neg_lanes = block.block_network.get_negative_lanes()
                self.np_random.shuffle(neg_lanes)
                trigger_lanes += neg_lanes
            potential_vehicle_configs = []
            for lanes in trigger_lanes:
                for l in lanes:
                    # if hasattr(self.engine, "object_manager") and l in self.engine.object_manager.accident_lanes:
                    #     continue
                    potential_vehicle_configs += self._propose_vehicle_configs(l)

            # How many vehicles should we spawn in this block?
            # total_length = sum([lane.length for lanes in trigger_lanes for lane in lanes])
            # total_spawn_points = int(math.floor(total_length / self.VEHICLE_GAP))
            # total_vehicles = int(math.floor(total_spawn_points * traffic_density))

            # Generate vehicles!
            # vehicles_on_block = []
            self.np_random.shuffle(potential_vehicle_configs)
            selected = potential_vehicle_configs[:min(total_vehicles, len(potential_vehicle_configs))]
            # print("We have {} candidates! We are spawning {} vehicles!".format(total_vehicles, len(selected)))

            from metadrive.policy.idm_policy import IDMPolicy
            from metadrive.policy.expert_policy import ExpertPolicy
            # for v_config in selected:
            v_config_i = random.randint(0, len(selected)-1)
            v_config = selected[v_config_i]
            lane_index = v_config["spawn_lane_index"]
            if lane_index[0][0] == '-':
                use_neg = True
            vehicle_type = self.random_vehicle_type()
            v_config.update(self.engine.global_config["traffic_vehicle_config"])
            #  lidar=dict(num_lasers=72, distance=40, num_others=0),
            v_config["lidar"] = dict(num_lasers=72, distance=40, num_others=0)
            random_v = self.spawn_object(vehicle_type, vehicle_config=v_config)
            self._traffic_vehicles.append(random_v)

            self.add_policy(random_v.id, AdvPolicy, random_v, self.generate_seed())

            # vehicles_on_block.append(random_v.name)

        #     trigger_road = block.pre_block_socket.positive_road if not use_neg else block.pre_block_socket.negative_road
        #     block_vehicles = BlockVehicles(trigger_road=trigger_road, vehicles=vehicles_on_block)
        #
        #     self.block_triggered_vehicles.append(block_vehicles)
        #     vehicle_num += len(vehicles_on_block)
        # self.block_triggered_vehicles.reverse()



    def random_vehicle_type(self):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type
        vehicle_type = random_vehicle_type(self.np_random, [0.2, 0.3, 0.3, 0.2, 0.0])
        return vehicle_type




# For compatibility check
# TrafficManager = PGAdversaryVechileManager


