from metadrive.component.map.pg_map import PGMap
from metadrive.component.pg_space import Parameter
from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.roundabout import Roundabout
from metadrive.component.pgblock.straight import Straight
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
        FirstPGBlock.ENTRANCE_LENGTH = 0.5
        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"

        # test = TestBlock(False)
        # initialize_asset_loader(engine=test)
        # global_network = NodeRoadNetwork()
        blocks = []
        init_block = FirstPGBlock(self.road_network, 3.0, 3, parent_node_path, physics_world, 1)
        self.blocks.append(init_block)

        block_s1 = Straight(1, init_block.get_socket(0), self.road_network, 1)
        block_s1.construct_from_config(
            {
                Parameter.length: 100
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_s1)

        block_c1 = Curve(2, block_s1.get_socket(0), self.road_network, 1)
        block_c1.construct_from_config({
            Parameter.length: 200,
            Parameter.radius: 100,
            Parameter.angle: 90,
            Parameter.dir: 1,
        }, parent_node_path, physics_world)
        self.blocks.append(block_c1)

        block_s2 = Straight(3, block_c1.get_socket(0), self.road_network, 1)
        block_s2.construct_from_config(
            {
                Parameter.length: 100,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_s2)

        block_c2 = Curve(4, block_s2.get_socket(0), self.road_network, 1)
        block_c2.construct_from_config({
            Parameter.length: 100,
            Parameter.radius: 60,
            Parameter.angle: 90,
            Parameter.dir: 1,
        }, parent_node_path, physics_world)
        self.blocks.append(block_c2)

        block_c3 = Curve(5, block_c2.get_socket(0), self.road_network, 1)
        block_c3.construct_from_config({
            Parameter.length: 100,
            Parameter.radius: 60,
            Parameter.angle: 90,
            Parameter.dir: 1,
        }, parent_node_path, physics_world)
        self.blocks.append(block_c3)

        block_s3 = Straight(6, block_c3.get_socket(0), self.road_network, 1)
        block_s3.construct_from_config(
            {
                Parameter.length: 200,
            }, parent_node_path, physics_world
        )
        self.blocks.append(block_s3)

        block_c4 = Curve(7, block_s3.get_socket(0), self.road_network, 1)
        block_c4.construct_from_config({
            Parameter.length: 100,
            Parameter.radius: 60,
            Parameter.angle: 90,
            Parameter.dir: 1,
        }, parent_node_path, physics_world)
        self.blocks.append(block_c4)


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
    Racing_config = dict(
        # controller="joystick",
        # num_agents=2,
        use_render=False,
        manual_control=True,
        traffic_density=0.1,
        random_agent_model=False,
        top_down_camera_initial_x=95,
        top_down_camera_initial_y=15,
        top_down_camera_initial_z=120,
        # random_lane_width=True,
        # random_lane_num=True,
        on_continuous_line_done=False,
        out_of_route_done=True,
        vehicle_config=dict(show_lidar=False, show_navi_mark=False),
    )

    # MABidirectionConfig = dict(
    #     spawn_roads=[Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3), -Road(Split.node(3, 0, 0), Split.node(3, 0, 1))],
    #     num_agents=20,
    #     map_config=dict(exit_length=60, bottle_lane_num=4, neck_lane_num=1, neck_length=20),
    #     top_down_camera_initial_x=95,
    #     top_down_camera_initial_y=15,
    #     top_down_camera_initial_z=120,
    #     cross_yellow_line_done=True,
    #     vehicle_config={
    #         "show_lidar": False,
    #         # "show_side_detector": True,
    #         # "show_lane_line_detector": True,
    #         "side_detector": dict(num_lasers=4, distance=50),  # laser num, distance
    #         "lane_line_detector": dict(num_lasers=4, distance=20)
    #     }  # laser num, distance
    # )
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    env = RacingEnv(Racing_config)
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
