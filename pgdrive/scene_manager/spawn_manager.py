import copy
from math import floor

import numpy as np
from panda3d.bullet import BulletBoxShape, BulletGhostNode
from panda3d.core import TransformState
from panda3d.core import Vec3, BitMask32

from pgdrive.constants import CollisionGroup
from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.utils import get_np_random
from pgdrive.utils.coordinates_shift import panda_position, panda_heading
from pgdrive.world.pg_world import PGWorld
from pgdrive.utils.scene_utils import rect_region_detection


class SpawnManager:
    """
    This class maintain a list of possible spawn places.
    """
    FORCE_AGENT_NAME = "force_agent_name"
    REGION_DETECT_HEIGHT = 10
    REGION_DETECT_LONGITUDE = 8
    REGION_DETECT_LATERAL = 3

    def __init__(self, exit_length, lane_num, num_agents, vehicle_config, target_vehicle_configs=None):
        self.num_agents = num_agents
        self.exit_length = (exit_length - FirstBlock.ENTRANCE_LENGTH)
        self.lane_num = lane_num
        self.vehicle_config = vehicle_config
        self.spawn_roads = []
        self.target_vehicle_configs = []
        self.safe_spawn_places = {}
        self.need_update_spawn_places = True
        self.initialized = False
        self.target_vehicle_configs = target_vehicle_configs
        self.spawn_places_used = []

        if self.num_agents is None:
            assert not self.target_vehicle_configs, (
                "You should now specify config if requiring infinite number of vehicles."
            )

    def set_spawn_roads(self, spawn_roads):
        if self.target_vehicle_configs:
            target_vehicle_configs, safe_spawn_places = self._update_spawn_roads_with_configs(spawn_roads)
        else:
            target_vehicle_configs, safe_spawn_places = self._update_spawn_roads_randomly(spawn_roads)
        self.target_vehicle_configs = target_vehicle_configs
        self.safe_spawn_places = {place["identifier"]: place for place in safe_spawn_places}
        self.spawn_roads = spawn_roads
        self.need_update_spawn_places = True
        self.initialized = True

    def _update_spawn_roads_with_configs(self, spawn_roads=None):
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
        if self.num_agents is not None:
            assert self.num_agents > 0
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

    def get_target_vehicle_configs(self, seed=None):
        num_agents = self.num_agents if self.num_agents is not None else len(self.target_vehicle_configs)
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

    def step(self):
        self.spawn_places_used = []

    def get_available_spawn_places(self, pg_world: PGWorld, map):
        ret = {}
        for bid, bp in self.safe_spawn_places.items():
            if bid in self.spawn_places_used:
                continue
            # save time
            if not bp.get("spawn_point_position", False):
                lane = map.road_network.get_lane(bp["config"]["spawn_lane_index"])
                long = bp["config"]["spawn_longitude"]
                lat = bp["config"]["spawn_lateral"]
                spawn_point_position = lane.position(longitudinal=long, lateral=lat)
                bp["spawn_point_position"] = (spawn_point_position[0], spawn_point_position[1])
                bp["spawn_point_heading"] = np.rad2deg(lane.heading_at(long))

            spawn_point_position = bp["spawn_point_position"]
            lane_heading = bp["spawn_point_heading"]
            result = rect_region_detection(
                pg_world, spawn_point_position, lane_heading, self.REGION_DETECT_LONGITUDE, self.REGION_DETECT_LATERAL,
                CollisionGroup.EgoVehicle
            )
            if (pg_world.world_config["debug"] or pg_world.world_config["debug_physics_world"]) and bp.get("need_debug",
                                                                                                           True):
                shape = BulletBoxShape(Vec3(self.REGION_DETECT_LONGITUDE / 2, self.REGION_DETECT_LATERAL / 2, 1))
                vis_body = pg_world.render.attach_new_node(BulletGhostNode("debug"))
                vis_body.node().addShape(shape)
                vis_body.setH(panda_heading(lane_heading))
                vis_body.setPos(panda_position(spawn_point_position, z=2))
                pg_world.physics_world.dynamic_world.attach(vis_body.node())
                vis_body.node().setIntoCollideMask(BitMask32.allOff())
                bp["need_debug"] = False

            if not result.hasHit():
                ret[bid] = bp
                self.spawn_places_used.append(bid)
            # elif pg_world.world_config["debug"] or pg_world.world_config["debug_physics_world"]:
            #     print(result.getNode())
        return ret
