import copy
from math import floor

import numpy as np

from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.utils import get_np_random, distance_greater


class SpawnManager:
    """
    This class maintain a list of possible spawn places.
    """
    FORCE_AGENT_NAME = "force_agent_name"

    def __init__(self, exit_length, lane_num, num_agents, vehicle_config, target_vehicle_configs=None):
        self.num_agents = num_agents
        self.exit_length = (exit_length - FirstBlock.ENTRANCE_LENGTH)
        self.lane_num = lane_num
        self.vehicle_config = vehicle_config
        self.spawn_roads = []
        self.target_vehicle_configs = []
        self.safe_spawn_places = {}
        self.mapping = {}
        self.need_update_spawn_places = True
        self.initialized = False
        self.target_vehicle_configs = target_vehicle_configs

    def update_spawn_roads(self, spawn_roads):
        if self.target_vehicle_configs:
            target_vehicle_configs, safe_spawn_places = self._update_spawn_roads_with_configs(spawn_roads)
        else:
            target_vehicle_configs, safe_spawn_places = self._update_spawn_roads_randomly(spawn_roads)
        self.target_vehicle_configs = target_vehicle_configs
        self.safe_spawn_places = {v["identifier"]: v for v in safe_spawn_places}
        self.mapping = {i: set() for i in self.safe_spawn_places.keys()}
        self.spawn_roads = spawn_roads
        self.need_update_spawn_places = True
        self.initialized = True

    def _update_spawn_roads_with_configs(self, spawn_roads):
        assert self.num_agents <= len(self.target_vehicle_configs), (
            "Too many agents! We only accept {} agents, which is specified by the number of configs in "
            "target_vehicle_configs, but you have {} agents! "
            "You should require less agent or not to specify the target_vehicle_configs!".format(
                len(self.target_vehicle_configs), self.num_agents
            )
        )
        target_vehicle_configs = []
        safe_spawn_places = []
        for v_id, v_config in self.target_vehicle_configs.items():
            lane_tuple = v_config["spawn_lane_index"]
            target_vehicle_configs.append(
                dict(identifier="|".join((str(s) for s in lane_tuple)), config=v_config, force_agent_name=v_id)
            )
            safe_spawn_places.append(target_vehicle_configs[-1].copy())
        return target_vehicle_configs, safe_spawn_places

    def _update_spawn_roads_randomly(self, spawn_roads):
        assert len(spawn_roads) > 0
        interval = 10
        num_slots = int(floor(self.exit_length / interval))
        interval = self.exit_length / num_slots
        assert self.num_agents <= self.lane_num * len(spawn_roads) * num_slots, (
            "Too many agents! We only accepet {} agents, but you have {} agents!".format(
                self.lane_num * len(spawn_roads) * num_slots, self.num_agents
            )
        )

        # We can spawn agents in the middle of road at the initial time, but when some vehicles need to be respawn,
        # then we have to set it to the farthest places to ensure safety (otherwise the new vehicles may suddenly
        # appear at the middle of the road!)
        target_vehicle_configs = []
        safe_spawn_places = []
        for i, road in enumerate(spawn_roads):
            for lane_idx in range(self.lane_num):
                for j in range(num_slots):
                    long = j * interval + np.random.uniform(0, 0.5 * interval)
                    lane_tuple = road.lane_index(lane_idx)  # like (>>>, 1C0_0_, 1) and so on.
                    target_vehicle_configs.append(
                        dict(
                            identifier="|".join((str(s) for s in lane_tuple + (j, ))),
                            config={
                                "spawn_lane_index": lane_tuple,
                                "spawn_longitude": long,
                                "spawn_lateral": self.vehicle_config["spawn_lateral"]
                            },
                            force_agent_name=None
                        )
                    )
                    if j == 0:
                        safe_spawn_places.append(target_vehicle_configs[-1].copy())
        return target_vehicle_configs, safe_spawn_places

    def get_target_vehicle_configs(self, num_agents, seed=None):
        assert len(self.target_vehicle_configs) > 0
        target_agents = get_np_random(seed).choice(
            [i for i in range(len(self.target_vehicle_configs))], num_agents, replace=False
        )

        # for rllib compatibility
        ret = {}
        if len(target_agents) > 1:
            for real_idx, idx in enumerate(target_agents):
                v_config = self.target_vehicle_configs[idx]["config"]
                ret["agent{}".format(real_idx)] = v_config
        else:
            ret["agent0"] = self.target_vehicle_configs[0]["config"]
        return copy.deepcopy(ret)

    def update(self, vehicles: dict, map):
        if self.need_update_spawn_places:
            assert self.initialized
            self.need_update_spawn_places = False
            for bid, bp in self.safe_spawn_places.items():
                lane = map.road_network.get_lane(bp["config"]["spawn_lane_index"])
                self.safe_spawn_places[bid]["position"] = lane.position(
                    longitudinal=bp["config"]["spawn_longitude"], lateral=bp["config"]["spawn_lateral"]
                )
                for vid in vehicles.keys():
                    self.confirm_respawn(bid, vid)  # Just assume everyone is all in the same spawn place at t=0.

        for bid, vid_set in self.mapping.items():
            removes = []
            for vid in vid_set:
                if (vid not in vehicles) or (distance_greater(self.safe_spawn_places[bid]["position"],
                                                              vehicles[vid].position, length=10)):
                    removes.append(vid)
            for vid in removes:
                self.mapping[bid].remove(vid)

    def confirm_respawn(self, spawn_place_id, vehicle_id):
        self.mapping[spawn_place_id].add(vehicle_id)

    def get_available_spawn_places(self):
        ret = {}
        for bid in self.safe_spawn_places.keys():
            if not self.mapping[bid]:  # empty
                ret[bid] = self.safe_spawn_places[bid]
        return ret
