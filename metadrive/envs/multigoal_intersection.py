"""
This file implement an intersection environment with multiple goals.
"""
from metadrive.constants import DEFAULT_AGENT
from metadrive.engine.logger import get_logger
from metadrive.manager.base_manager import BaseAgentManager
from metadrive.policy.AI_protect_policy import AIProtectPolicy
from metadrive.policy.idm_policy import TrajectoryIDMPolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy, TakeoverPolicy, TakeoverPolicyWithoutBrake
from metadrive.policy.replay_policy import ReplayTrafficParticipantPolicy
from metadrive.component.pgblock.intersection import InterSection
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.road_network import Road
import copy
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from typing import Union
import seaborn as sns

import numpy as np

from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import parse_map_config, MapGenerateMethod
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.envs.base_env import BaseEnv
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.utils import clip, Config



from metadrive import MetaDriveEnv

from metadrive.utils import clip
import copy

from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.intersection import InterSection
from metadrive.component.road_network import Road
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.manager.base_manager import BaseManager

from metadrive.manager.spawn_manager import SpawnManager
from metadrive.utils import Config
#
# MAIntersectionConfig = dict(
#     spawn_roads=[
#         Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3),
#         -Road(InterSection.node(1, 0, 0), InterSection.node(1, 0, 1)),
#         -Road(InterSection.node(1, 1, 0), InterSection.node(1, 1, 1)),
#         -Road(InterSection.node(1, 2, 0), InterSection.node(1, 2, 1)),
#     ],
#     num_agents=30,
#     map_config=dict(exit_length=60, lane_num=2),
#     top_down_camera_initial_x=80,
#     top_down_camera_initial_y=0,
#     top_down_camera_initial_z=120
# )





logger = get_logger()


class MultiGoalIntersectionNavigationManager(BaseManager):

    GOALS = {
        # TODO: Double check the meanings
        "u_turn": Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3),
        "left_turn": -Road(InterSection.node(1, 0, 0), InterSection.node(1, 0, 1)),
        "right_turn": -Road(InterSection.node(1, 1, 0), InterSection.node(1, 1, 1)),
        "go_straight": -Road(InterSection.node(1, 2, 0), InterSection.node(1, 2, 1)),
    }

    def __init__(self):
        super().__init__()

        config = self.engine.global_config
        vehicle_config = config["vehicle_config"]
        self.navigations = {}
        navi = NodeNetworkNavigation

        colors = sns.color_palette("colorblind")
        for c, (dest_name, road) in enumerate(self.GOALS.items()):
            self.navigations[dest_name] = navi(
                # self.engine,
                show_navi_mark=vehicle_config["show_navi_mark"],
                show_dest_mark=vehicle_config["show_dest_mark"],
                show_line_to_dest=vehicle_config["show_line_to_dest"],
                panda_color=colors[c],  # color for navigation marker
                name=dest_name,
                vehicle_config=vehicle_config
            )

    @property
    def agent(self):
        return self.engine.agents[DEFAULT_AGENT]

    @property
    def goals(self):
        return self.GOALS

    def after_reset(self):
        print("[DEBUG]: after_reset in MultiGoalIntersectionNavigationManager")
        for name, navi in self.navigations.items():
            navi.reset(self.agent, dest=self.GOALS[name])
            navi.update_localization(self.agent)

    def after_step(self):
        print("[DEBUG]: after_step in MultiGoalIntersectionNavigationManager")
        for name, navi in self.navigations.items():
            navi.update_localization(self.agent)

    def get_navigation(self, goal_name):
        assert goal_name in self.GOALS, "Invalid goal name!"
        return self.navigations[goal_name]










class MultiGoalIntersectionMap(PGMap):
    """This class does nothing but turn on the U-turn."""
    def _generate(self):
        length = self.config["exit_length"]
        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"
        # Build a first-block
        last_block = FirstPGBlock(
            self.road_network,
            self.config[self.LANE_WIDTH],
            self.config[self.LANE_NUM],
            parent_node_path,
            physics_world,
            length=length
        )
        self.blocks.append(last_block)
        # Build Intersection
        InterSection.EXIT_PART_LENGTH = length
        if "radius" in self.config and self.config["radius"]:
            extra_kwargs = dict(radius=self.config["radius"])
        else:
            extra_kwargs = {}
        last_block = InterSection(
            1,
            last_block.get_socket(index=0),
            self.road_network,
            random_seed=1,
            ignore_intersection_checking=False,
            **extra_kwargs
        )
        assert self.config["lane_num"] > 1
        last_block.enable_u_turn(True)  # <<<<<<<<<< We turn on U turn here
        last_block.construct_block(parent_node_path, physics_world)
        self.blocks.append(last_block)


class MultiGoalPGMapManager(PGMapManager):
    """This class simply does nothing but load MultiGoalIntersectionMap."""
    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(MultiGoalIntersectionMap, map_config=config["map_config"], random_seed=None)
        else:
            assert len(self.spawned_objects) == 1, "It is supposed to contain one map in this manager"
            _map = self.spawned_objects.values()[0]
        self.load_map(_map)
        # self.current_map.spawn_roads = config["spawn_roads"]


class MultiGoalIntersectionEnv(MetaDriveEnv):
    @classmethod
    def default_config(cls):
        config = MetaDriveEnv.default_config()
        config.update(
            {

                # Set the map to an Intersection
                "start_seed": 0,
                "num_scenarios": 1,
                "map": "X",

                "vehicle_config": {

                    # Turn off vehicle's own navigation module.
                    "navigation_module": None,


                    "side_detector": dict(num_lasers=4, distance=50),  # laser num, distance

                    # To avoid goal-dependent lane detection, we use Lidar to detect distance to nearby lane lines.
                    # Otherwise, we will ask the navigation module to provide current lane and extract the lateral
                    # distance directly on this lane.
                    "lane_line_detector": dict(num_lasers=4, distance=20)
                }

            }
        )
        return config

    def setup_engine(self):
        super().setup_engine()

        # We use MultiGoalPGMapManager here
        self.engine.update_manager("map_manager", MultiGoalPGMapManager())

        # Introducing a new navigation manager
        self.engine.register_manager("goal_manager", MultiGoalIntersectionNavigationManager())


    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        reward = 0.0
        vehicle = self.agents[vehicle_id]
        step_info = dict()


        for goal_name in self.engine.goal_manager.goals.keys():

            # Reward for moving forward in current lane
            # TODO: Remove this part for now.
            # if vehicle.lane in vehicle.navigation.current_ref_lanes:
            #     current_lane = vehicle.lane
            #     positive_road = 1
            # else:
            #     current_lane = vehicle.navigation.current_ref_lanes[0]
            #     current_road = vehicle.navigation.current_road
            #     positive_road = 1 if not current_road.is_negative_road() else -1
            # long_last, _ = current_lane.local_coordinates(vehicle.last_position)
            # long_now, lateral_now = current_lane.local_coordinates(vehicle.position)
            # reward += self.config["driving_reward"] * (long_now - long_last) * positive_road
            # reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) * positive_road


            step_info["step_reward"] = reward
            if self._is_arrive_destination(vehicle, goal_name):
                reward = +self.config["success_reward"]
            elif self._is_out_of_road(vehicle):
                reward = -self.config["out_of_road_penalty"]
            elif vehicle.crash_vehicle:
                reward = -self.config["crash_vehicle_penalty"]
            elif vehicle.crash_object:
                reward = -self.config["crash_object_penalty"]
            step_info["route_completion"] = vehicle.navigation.route_completion
            # TODO Finish this part! =======================
            # TODO Finish this part! =======================
            # TODO Finish this part! =======================
            # TODO Finish this part! =======================
            # TODO Finish this part! =======================
            # TODO Finish this part! =======================
            # TODO Finish this part! =======================
            # TODO Finish this part! =======================
            # return reward, step_info

    def _is_arrive_destination(self, vehicle, goal_name):
        """
        Args:
            vehicle: The BaseVehicle instance.
            goal_name: The name of the goal.

        Returns:
            flag: Whether this vehicle arrives its destination.
        """
        navi = self.engine.goal_manager.get_navigation(goal_name)
        long, lat = navi.final_lane.local_coordinates(vehicle.position)
        flag = (navi.final_lane.length - 5 < long < navi.final_lane.length + 5) and (
            navi.get_current_lane_width() / 2 >= lat >=
            (0.5 - navi.get_current_lane_num()) * navi.get_current_lane_width()
        )
        return flag


if __name__ == "__main__":
    config = dict(
        use_render=True,
        manual_control=True,
        vehicle_config=dict(show_lidar=False, show_navi_mark=True, show_line_to_navi_mark=True),
    )
    env = MultiGoalIntersectionEnv(config)
    try:
        o, _ = env.reset()
        for i in range(1, 1000000000):
            o, r, tm, tc, info = env.step([0, 0])
            env.render()
            env.render(mode="topdown")
            if (tm or tc) and info["arrive_dest"]:
                env.reset()
                # env.current_track_agent.expert_takeover = True
    finally:
        env.close()
