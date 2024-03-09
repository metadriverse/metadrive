import logging
from collections import namedtuple
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.component.map.base_map import BaseMap
from metadrive.manager.traffic_manager import PGTrafficManager
BlockVehicles = namedtuple("block_vehicles", "trigger_road vehicles")


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
        if abs(traffic_density) < 1e-2:
            return
        self.respawn_lanes = self._get_available_respawn_lanes(map)
        if self.mode == TrafficMode.Respawn:
            # add respawn vehicle
            self._create_respawn_vehicles(map, traffic_density)
        elif self.mode == TrafficMode.Trigger or self.mode == TrafficMode.Hybrid:
            self._create_vehicles_once(map, traffic_density)
        elif self.mode == TrafficMode.Adversary: # spawn the adversary vehicles
            self._create_adversary_vehicles(map, self.num_adversary_vehicles)
        else:
            raise ValueError("No such mode named {}".format(self.mode))



    def get_adv_config(self, lane, long): # TODO: used for spawn
        return {"spawn_lane_index": lane.index, "spawn_longitude": long, "enable_reverse": False}


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
            adv_vehicle_config = self.engine.global_config["agent_configs"]["default_agent"] # in the single agent case, this is the same as agent0

            # adv_vehicle_config.update(self.engine.global_config["traffic_vehicle_config"]) # TODO: not necessary??
            random_v = self.spawn_object(vehicle_type, vehicle_config=adv_vehicle_config)
            self._traffic_vehicles.append(random_v)

            from metadrive.policy.adv_policy import AdvPolicy
            self.add_policy(random_v.id, AdvPolicy, random_v, self.generate_seed())
            vehicles_on_block.append(random_v.name)

        #TODO: copied  below block lines from _create_vehivle_once(), what does this do? maybe we should remove this.....
        trigger_road = first_real_block.pre_block_socket.positive_road
        block_vehicles = BlockVehicles(trigger_road=trigger_road, vehicles=vehicles_on_block)
        self.block_triggered_vehicles.append(block_vehicles)
        vehicle_num += len(vehicles_on_block)
        self.block_triggered_vehicles.reverse()


    def random_vehicle_type(self):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type
        vehicle_type = random_vehicle_type(self.np_random, [0.2, 0.3, 0.3, 0.2, 0.0])
        return vehicle_type




# For compatibility check
# TrafficManager = PGAdversaryVechileManager


