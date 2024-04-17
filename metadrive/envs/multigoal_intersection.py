"""
This file implement an intersection environment with multiple goals.
"""
from collections import defaultdict

import numpy as np
import seaborn as sns

from metadrive import MetaDriveEnv
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.component.pg_space import ParameterSpace, Parameter, ConstantSpace, DiscreteSpace
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.intersection import InterSectionWithUTurn
from metadrive.component.road_network import Road
from metadrive.constants import DEFAULT_AGENT
from metadrive.engine.logger import get_logger
from metadrive.envs.varying_dynamics_env import VaryingDynamicsAgentManager, VaryingDynamicsConfig
from metadrive.manager.base_manager import BaseManager

logger = get_logger()


class CustomizedIntersection(InterSectionWithUTurn):
    PARAMETER_SPACE = ParameterSpace(
        {

            # changed from 10 to 8:
            Parameter.radius: ConstantSpace(8),

            # unchanged:
            Parameter.change_lane_num: DiscreteSpace(min=0, max=1),
            Parameter.decrease_increase: DiscreteSpace(min=0, max=1)
        }
    )


class MultiGoalIntersectionNavigationManager(BaseManager):
    """
    This manager is responsible for managing multiple navigation modules, each of which is responsible for guiding the
    agent to a specific goal.
    """
    GOALS = {
        "u_turn": (-Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3)).end_node,
        "right_turn": Road(
            CustomizedIntersection.node(block_idx=1, part_idx=0, road_idx=0),
            CustomizedIntersection.node(block_idx=1, part_idx=0, road_idx=1)
        ).end_node,
        "go_straight": Road(
            CustomizedIntersection.node(block_idx=1, part_idx=1, road_idx=0),
            CustomizedIntersection.node(block_idx=1, part_idx=1, road_idx=1)
        ).end_node,
        "left_turn": Road(
            CustomizedIntersection.node(block_idx=1, part_idx=2, road_idx=0),
            CustomizedIntersection.node(block_idx=1, part_idx=2, road_idx=1)
        ).end_node,
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
        """Reset all navigation modules."""
        # print("[DEBUG]: after_reset in MultiGoalIntersectionNavigationManager")
        for name, navi in self.navigations.items():
            navi.reset(self.agent, dest=self.goals[name])
            navi.update_localization(self.agent)

    def after_step(self):
        """Update all navigation modules."""
        # print("[DEBUG]: after_step in MultiGoalIntersectionNavigationManager")
        for name, navi in self.navigations.items():
            navi.update_localization(self.agent)

    def get_navigation(self, goal_name):
        """Return the navigation module for the given goal."""
        assert goal_name in self.goals, "Invalid goal name!"
        return self.navigations[goal_name]


class MultiGoalIntersectionEnv(MetaDriveEnv):
    """
    This environment is an intersection with multiple goals. We provide the reward function, observation, termination
    conditions for each goal in the info dict returned by env.reset and env.step, with prefix "goals/{goal_name}/".
    """
    @classmethod
    def default_config(cls):
        config = MetaDriveEnv.default_config()
        config.update(VaryingDynamicsConfig)
        config.update(
            {

                # Set the map to an Intersection
                "start_seed": 0,

                # Disable the shortcut config for map.
                "map": None,
                "map_config": dict(
                    type="block_sequence", config=[
                        CustomizedIntersection,
                    ], lane_num=2, lane_width=3.5
                ),

                # Even though the map will not change, the traffic flow will change.
                "num_scenarios": 1000,

                # Remove all traffic vehicles for now.
                "traffic_density": 0.2,
                "vehicle_config": {

                    # Remove navigation arrows in the window as we are in multi-goal environment.
                    "show_navigation_arrow": False,

                    # Turn off vehicle's own navigation module.
                    "side_detector": dict(num_lasers=4, distance=50),  # laser num, distance

                    # To avoid goal-dependent lane detection, we use Lidar to detect distance to nearby lane lines.
                    # Otherwise, we will ask the navigation module to provide current lane and extract the lateral
                    # distance directly on this lane.
                    "lane_line_detector": dict(num_lasers=4, distance=20)
                }
            }
        )
        return config

    def _get_agent_manager(self):
        return VaryingDynamicsAgentManager(init_observations=self._get_observations())

    def setup_engine(self):
        super().setup_engine()

        # Introducing a new navigation manager
        self.engine.register_manager("goal_manager", MultiGoalIntersectionNavigationManager())

    def _get_step_return(self, actions, engine_info):
        """Add goal-dependent observation to the info dict."""
        o, r, tm, tc, i = super(MultiGoalIntersectionEnv, self)._get_step_return(actions, engine_info)
        for goal_name in self.engine.goal_manager.goals.keys():
            navi = self.engine.goal_manager.get_navigation(goal_name)
            navi_info = navi.get_navi_info()
            i["goals/{}/obs".format(goal_name)] = np.asarray(navi_info).astype(np.float32)
        return o, r, tm, tc, i

    def _get_reset_return(self, reset_info):
        """Add goal-dependent observation to the info dict."""
        o, i = super(MultiGoalIntersectionEnv, self)._get_reset_return(reset_info)
        for goal_name in self.engine.goal_manager.goals.keys():
            navi = self.engine.goal_manager.get_navigation(goal_name)
            navi_info = navi.get_navi_info()
            i["goals/{}/obs".format(goal_name)] = np.asarray(navi_info).astype(np.float32)
        return o, i

    def reward_function(self, vehicle_id: str):
        """
        Compared to the original reward_function, we add goal-dependent reward to info dict.
        """
        vehicle = self.agents[vehicle_id]
        step_info = dict()

        # Compute goal-agnostic reward
        goal_agnostic_reward = 0.0
        goal_agnostic_reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h)

        # Can't assign arrive-destination reward if goal is not known.
        # if self._is_arrive_destination(vehicle, goal_name):
        #     reward = +self.config["success_reward"]
        if self._is_out_of_road(vehicle):
            goal_agnostic_reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            goal_agnostic_reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            goal_agnostic_reward = -self.config["crash_object_penalty"]
        step_info["step_reward"] = goal_agnostic_reward
        step_info[f"route_completion"] = float("nan")  # Can't report this value if goal is not known.

        # Compute goal-dependent reward and saved to step_info
        for goal_name in self.engine.goal_manager.goals.keys():
            navi = self.engine.goal_manager.get_navigation(goal_name)
            prefix = "goals/{}".format(goal_name)
            reward = 0.0

            # Get goal-dependent information
            if navi.current_lane in navi.current_ref_lanes:
                current_lane = navi.current_lane
                positive_road = 1
            else:
                current_lane = navi.current_ref_lanes[0]
                current_road = navi.current_road
                positive_road = 1 if not current_road.is_negative_road() else -1
            long_last, _ = current_lane.local_coordinates(vehicle.last_position)
            long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

            # Reward for moving forward in current lane
            reward += self.config["driving_reward"] * (long_now - long_last) * positive_road

            # Reward for speed, sign determined by whether in the correct lanes (instead of driving in the wrong
            # direction).
            reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) * positive_road
            if self._is_arrive_destination(vehicle, goal_name):
                reward = +self.config["success_reward"]
            elif self._is_out_of_road(vehicle):
                reward = -self.config["out_of_road_penalty"]
            elif vehicle.crash_vehicle:
                reward = -self.config["crash_vehicle_penalty"]
            elif vehicle.crash_object:
                reward = -self.config["crash_object_penalty"]
            step_info[f"{prefix}/step_reward"] = reward
            step_info[f"{prefix}/route_completion"] = navi.route_completion

        return goal_agnostic_reward, step_info

    def _is_arrive_destination(self, vehicle, goal_name=None):
        """
        Compared to the original function, here we look up the navigation from goal_manager.

        Args:
            vehicle: The BaseVehicle instance.
            goal_name: The name of the goal. If None, return True if any goal is arrived.

        Returns:
            flag: Whether this vehicle arrives its destination.
        """
        if goal_name is None:
            ret = False
            for name in self.engine.goal_manager.goals.keys():
                ret = ret or self._is_arrive_destination(vehicle, name)
            return ret

        navi = self.engine.goal_manager.get_navigation(goal_name)
        long, lat = navi.final_lane.local_coordinates(vehicle.position)
        flag = (navi.final_lane.length - 5 < long < navi.final_lane.length + 5) and (
            navi.get_current_lane_width() / 2 >= lat >=
            (0.5 - navi.get_current_lane_num()) * navi.get_current_lane_width()
        )
        return flag

    def done_function(self, vehicle_id: str):
        """
        Compared to MetaDriveEnv's done_function, we add more stats here to record which goal is arrived.
        """
        done, done_info = super(MultiGoalIntersectionEnv, self).done_function(vehicle_id)
        vehicle = self.agents[vehicle_id]

        for goal_name in self.engine.goal_manager.goals.keys():
            prefix = "goals/{}".format(goal_name)
            done_info[f"{prefix}/arrive_dest"] = self._is_arrive_destination(vehicle, goal_name)

        return done, done_info


if __name__ == "__main__":
    config = dict(
        use_render=True,
        manual_control=True,
        vehicle_config=dict(show_lidar=False, show_navi_mark=True, show_line_to_navi_mark=True),
        accident_prob=1.0,
        decision_repeat=5,
    )
    env = MultiGoalIntersectionEnv(config)
    episode_rewards = defaultdict(float)
    try:
        o, info = env.reset()

        print('=======================')
        print("Goal-agnostic observation shape:\n\t", o.shape)
        print("Observation shape for each goals: ")
        for k in sorted(info.keys()):
            if k.startswith("goals/") and k.endswith("obs"):
                print(f"\t{k}: {info[k].shape}")
        print('=======================')

        s = 0
        for i in range(1, 1000000000):
            o, r, tm, tc, info = env.step([0, 1])
            done = tm or tc
            s += 1
            env.render()
            env.render(mode="topdown")

            for k, v in info.items():
                if k.startswith("goals/") and k.endswith("reward"):
                    episode_rewards[k] += v

            if s % 20 == 0:
                info = {k: info[k] for k in sorted(info.keys())}
                print('\n===== timestep {} ====='.format(s))
                for k, v in info.items():
                    if k.startswith("goals/") and not k.endswith('obs'):
                        print(f"{k}: {v:.2f}")
                print('=======================')

            if done:
                print('\n===== timestep {} ====='.format(s))
                print("EPISODE DONE\n")
                print('route completion:')
                for k in sorted(info.keys()):
                    kk = k.replace("/route_completion", "")
                    if k.startswith("goals/") and k.endswith("route_completion"):
                        print(f"\t{kk}: {info[k]:.2f}")

                print('\narrive destination (success):')
                for k in sorted(info.keys()):
                    kk = k.replace("/arrive_dest", "")
                    if k.startswith("goals/") and k.endswith("arrive_dest"):
                        print(f"\t{kk}: {info[k]:.2f}")

                print('\nepisode_rewards:')
                for k in sorted(episode_rewards.keys()):
                    kk = k.replace("/step_reward", "")
                    print(f"\t{kk}: {episode_rewards[k]:.2f}")
                episode_rewards.clear()
                print('=======================')

            if done:
                print('\n\n\n')
                env.reset()
                s = 0
    finally:
        env.close()
