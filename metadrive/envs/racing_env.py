from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.roundabout import Roundabout
from metadrive.component.road_network import Road
from metadrive.manager.pg_map_manager import PGMapManager
import copy
from metadrive import MetaDriveEnv
from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.roundabout import Roundabout
from metadrive.component.road_network import Road
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.manager.spawn_manager import SpawnManager
from metadrive.utils import Config


import argparse
import cv2
import numpy as np

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE



class RacingMap(PGMap):
    def _generate(self):
        super(RacingMap, self)._generate()

# class RoundaboutSpawnManager(SpawnManager):
#     def update_destination_for(self, vehicle_id, vehicle_config):
#         end_roads = copy.deepcopy(self.engine.global_config["spawn_roads"])
#         end_road = -self.np_random.choice(end_roads)  # Use negative road!
#         vehicle_config["destination"] = end_road.end_node
#         return vehicle_config
# 



class RacingMapManager(PGMapManager):
    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(RacingMap, map_config=config["map_config"], random_seed=None)
        else:
            assert len(self.spawned_objects) == 1, "It is supposed to contain one map in this manager"
            _map = self.spawned_objects.values()[0]
        self.load_map(_map)
        # self.current_map.spawn_roads = config["spawn_roads"]


class RacingEnv(MetaDriveEnv):
    # @staticmethod
    # def default_config() -> Config:
    #     return MultiAgentMetaDrive.default_config().update(MARoundaboutConfig, allow_add_new_key=True)

    def setup_engine(self):
        super(RacingEnv, self).setup_engine()
        # self.engine.update_manager("spawn_manager", RoundaboutSpawnManager())
        self.engine.update_manager("map_manager", RacingMapManager())

if __name__ == "__main__":
    config = dict(
        # controller="joystick",
        # num_agents=2,
        use_render=False,
        manual_control=True,
        traffic_density=0.1,
        num_scenarios=10000,
        random_agent_model=False,
        random_lane_width=True,
        random_lane_num=True,
        on_continuous_line_done=False,
        out_of_route_done=True,
        vehicle_config=dict(show_lidar=False, show_navi_mark=False),
        map_config={"config": "CCC", "type": "block_sequence"},
        start_seed=10,
    )
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    env = RacingEnv(config)
    try:
        o, _ = env.reset(seed=21)
        print(HELP_MESSAGE)
        env.vehicle.expert_takeover = True
        for i in range(1, 1000000000):
            o, r, tm, tc, info = env.step([0, 0])
            env.render(mode="topdown")
            if (tm or tc) and info["arrive_dest"]:
                env.reset(env.current_seed + 1)
                env.current_track_vehicle.expert_takeover = True
    except Exception as e:
        raise e
    finally:
        env.close()
