"""
This file provides a multi-goal environment based on the intersection environment. The environment fully support
conventional MetaDrive PG maps, where there is a special config['use_pg_map'] to enable the PG maps and all config are
the same as MetaDriveEnv.
If config['use_pg_map'] is False, the environment will use an intersection map and the goals information for all
possible destinations will be provided.
"""
from collections import defaultdict

import gymnasium as gym
import numpy as np

from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.component.pg_space import ParameterSpace, Parameter, DiscreteSpace, BoxSpace
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.intersection import InterSectionWithUTurn
from metadrive.component.road_network import Road
from metadrive.constants import DEFAULT_AGENT, get_color_palette
from metadrive.engine.logger import get_logger
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.manager.base_manager import BaseManager
from metadrive.obs.state_obs import BaseObservation, StateObservation
from metadrive.utils.math import clip, norm

logger = get_logger()

EGO_STATE_DIM = 5
NAVI_DIM = 10
GOAL_DEPENDENT_STATE_DIM = 3


class CustomizedObservation(BaseObservation):
    def __init__(self, config):
        self.state_obs = StateObservation(config)
        super(CustomizedObservation, self).__init__(config)
        self.latest_observation = {}

        self.lane_detect_dim = self.config['vehicle_config']['lane_line_detector']['num_lasers']
        self.side_detect_dim = self.config['vehicle_config']['side_detector']['num_lasers']
        self.vehicle_detect_dim = self.config['vehicle_config']['lidar']['num_lasers']

    @property
    def observation_space(self):
        shape = (
            EGO_STATE_DIM + self.side_detect_dim + self.lane_detect_dim + self.vehicle_detect_dim + NAVI_DIM +
            GOAL_DEPENDENT_STATE_DIM,
        )
        return gym.spaces.Box(-1.0, 1.0, shape=shape, dtype=np.float32)

    def observe(self, vehicle, navigation=None):
        ego = self.state_observe(vehicle)
        assert ego.shape[0] == EGO_STATE_DIM

        obs = [ego]

        if vehicle.config["side_detector"]["num_lasers"] > 0:
            side = self.side_detector_observe(vehicle)
            assert side.shape[0] == self.side_detect_dim
            obs.append(side)
            self.latest_observation["side_detect"] = side

        if vehicle.config["lane_line_detector"]["num_lasers"] > 0:
            lane = self.lane_line_detector_observe(vehicle)
            assert lane.shape[0] == self.lane_detect_dim
            obs.append(lane)
            self.latest_observation["lane_detect"] = lane

        if vehicle.config["lidar"]["num_lasers"] > 0:
            veh = self.vehicle_detector_observe(vehicle)
            assert veh.shape[0] == self.vehicle_detect_dim
            obs.append(veh)
            self.latest_observation["vehicle_detect"] = veh
        if navigation is None:
            navigation = vehicle.navigation
        navi = navigation.get_navi_info()
        assert len(navi) == NAVI_DIM
        obs.append(navi)

        # Goal-dependent infos
        goal_dependent_info = []
        lateral_to_left, lateral_to_right = vehicle._dist_to_route_left_right(navigation=navigation)
        if self.engine.current_map:
            total_width = float((self.engine.current_map.MAX_LANE_NUM + 1) * self.engine.current_map.MAX_LANE_WIDTH)
        else:
            total_width = 100
        lateral_to_left /= total_width
        lateral_to_right /= total_width
        goal_dependent_info += [clip(lateral_to_left, 0.0, 1.0), clip(lateral_to_right, 0.0, 1.0)]
        current_reference_lane = navigation.current_ref_lanes[-1]
        goal_dependent_info += [
            # The angular difference between vehicle's heading and the lane heading at this location.
            vehicle.heading_diff(current_reference_lane),
        ]
        goal_dependent_info = np.asarray(goal_dependent_info)
        assert goal_dependent_info.shape[0] == GOAL_DEPENDENT_STATE_DIM
        obs.append(goal_dependent_info)

        obs = np.concatenate(obs).astype(np.float32)

        self.latest_observation["state"] = ego
        self.latest_observation["raw_navi"] = navi

        return obs

    def state_observe(self, vehicle):
        # update out of road
        info = np.zeros([
            EGO_STATE_DIM,
        ])

        # The velocity of target vehicle
        info[0] = clip((vehicle.speed_km_h + 1) / (vehicle.max_speed_km_h + 1), 0.0, 1.0)

        # Current steering
        info[1] = clip((vehicle.steering / vehicle.MAX_STEERING + 1) / 2, 0.0, 1.0)

        # The normalized actions at last steps
        info[2] = clip((vehicle.last_current_action[1][0] + 1) / 2, 0.0, 1.0)
        info[3] = clip((vehicle.last_current_action[1][1] + 1) / 2, 0.0, 1.0)

        # Current angular acceleration (yaw rate)
        heading_dir_last = vehicle.last_heading_dir
        heading_dir_now = vehicle.heading
        cos_beta = heading_dir_now.dot(heading_dir_last) / (norm(*heading_dir_now) * norm(*heading_dir_last))
        beta_diff = np.arccos(clip(cos_beta, 0.0, 1.0))
        yaw_rate = beta_diff / 0.1
        info[4] = clip(yaw_rate, 0.0, 1.0)

        return info

    def side_detector_observe(self, vehicle):
        return np.asarray(
            self.engine.get_sensor("side_detector").perceive(
                vehicle,
                num_lasers=vehicle.config["side_detector"]["num_lasers"],
                distance=vehicle.config["side_detector"]["distance"],
                physics_world=vehicle.engine.physics_world.static_world,
                show=vehicle.config["show_side_detector"],
            ).cloud_points,
            dtype=np.float32
        )

    def lane_line_detector_observe(self, vehicle):
        return np.asarray(
            self.engine.get_sensor("lane_line_detector").perceive(
                vehicle,
                vehicle.engine.physics_world.static_world,
                num_lasers=vehicle.config["lane_line_detector"]["num_lasers"],
                distance=vehicle.config["lane_line_detector"]["distance"],
                show=vehicle.config["show_lane_line_detector"],
            ).cloud_points,
            dtype=np.float32
        )

    def vehicle_detector_observe(self, vehicle):
        cloud_points, detected_objects = self.engine.get_sensor("lidar").perceive(
            vehicle,
            physics_world=self.engine.physics_world.dynamic_world,
            num_lasers=vehicle.config["lidar"]["num_lasers"],
            distance=vehicle.config["lidar"]["distance"],
            show=vehicle.config["show_lidar"],
        )
        return np.asarray(cloud_points, dtype=np.float32)

    def destroy(self):
        """
        Clear allocated memory
        """
        self.state_obs.destroy()
        super(CustomizedObservation, self).destroy()
        self.cloud_points = None
        self.detected_objects = None


class CustomizedIntersection(InterSectionWithUTurn):
    PARAMETER_SPACE = ParameterSpace(
        {
            Parameter.radius: BoxSpace(min=9, max=20.0),
            Parameter.change_lane_num: DiscreteSpace(min=0, max=2),
            Parameter.decrease_increase: DiscreteSpace(min=0, max=0)
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
        colors = [get_color_palette()[c] for c in range(len(self.GOALS))]
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

    def after_step(self, *args, **kwargs):
        """Update all navigation modules."""
        # print("[DEBUG]: after_step in MultiGoalIntersectionNavigationManager")
        for name, navi in self.navigations.items():
            navi.update_localization(self.agent)
            # print("Navigation {} next checkpoint: {}".format(name, navi.get_checkpoints()))

    def get_navigation(self, goal_name):
        """Return the navigation module for the given goal."""
        assert goal_name in self.goals, "Invalid goal name!"
        return self.navigations[goal_name]


class MultiGoalIntersectionEnvBase(MetaDriveEnv):
    """
    This environment is an intersection with multiple goals. We provide the reward function, observation, termination
    conditions for each goal in the info dict returned by env.reset and env.step, with prefix "goals/{goal_name}/".
    """
    @classmethod
    def default_config(cls):
        config = MetaDriveEnv.default_config()
        # config.update(VaryingDynamicsConfig)
        config.update(
            {
                "use_multigoal_intersection": True,

                # Set the map to an Intersection
                "start_seed": 0,

                # Even though the map will not change, the traffic flow will change.
                "num_scenarios": 1000,

                # Remove all traffic vehicles for now.
                # "traffic_density": 0.2,

                # If the vehicle does not reach the default destination, it will receive a penalty.
                "wrong_way_penalty": 10.0,
                # "crash_sidewalk_penalty": 10.0,
                # "crash_vehicle_penalty": 10.0,
                # "crash_object_penalty": 10.0,
                # "out_of_road_penalty": 10.0,
                "out_of_route_penalty": 0.0,
                # "success_reward": 10.0,
                # "driving_reward": 1.0,
                # "on_continuous_line_done": True,
                # "out_of_road_done": True,
                "vehicle_config": {
                    "show_navi_mark": False,
                    "show_line_to_navi_mark": False,
                    "show_line_to_dest": False,
                    "show_dest_mark": False,

                    # Remove navigation arrows in the window as we are in multi-goal environment.
                    "show_navigation_arrow": False,

                    # Turn off vehicle's own navigation module.
                    "side_detector": dict(num_lasers=120, distance=50),  # laser num, distance
                    "lidar": dict(num_lasers=120, distance=50),

                    # To avoid goal-dependent lane detection, we use Lidar to detect distance to nearby lane lines.
                    # Otherwise, we will ask the navigation module to provide current lane and extract the lateral
                    # distance directly on this lane.
                    "lane_line_detector": dict(num_lasers=0, distance=20)
                }
            }
        )
        return config

    def _post_process_config(self, config):
        config = super()._post_process_config(config)
        if config["use_multigoal_intersection"]:
            config['map'] = None
            config['map_config'] = dict(
                type="block_sequence", config=[
                    CustomizedIntersection,
                ], lane_num=2, lane_width=3.5
            )
        return config

    # def _get_agent_manager(self):
    #     return VaryingDynamicsAgentManager(init_observations=self._get_observations())

    def get_single_observation(self):
        return CustomizedObservation(self.config)

    #     else:
    #         return super().get_single_observation()
    #         img_obs = self.config["image_observation"]
    #         o = ImageStateObservation(self.config) if img_obs else LidarStateObservation(self.config)

    def setup_engine(self):
        super().setup_engine()

        # Introducing a new navigation manager
        if self.config["use_multigoal_intersection"]:
            self.engine.register_manager("goal_manager", MultiGoalIntersectionNavigationManager())

    def _get_step_return(self, actions, engine_info):
        """Add goal-dependent observation to the info dict."""
        o, r, tm, tc, i = super(MultiGoalIntersectionEnvBase, self)._get_step_return(actions, engine_info)

        if self.config["use_multigoal_intersection"]:
            for goal_name in self.engine.goal_manager.goals.keys():
                navi = self.engine.goal_manager.get_navigation(goal_name)
                goal_obs = self.observations["default_agent"].observe(self.agents[DEFAULT_AGENT], navi)
                i["obs/goals/{}".format(goal_name)] = goal_obs
            assert r == i["reward/default_reward"]
            assert i["route_completion"] == i["route_completion/goals/default"]

        else:
            i["obs/goals/default"] = self.observations["default_agent"].observe(self.agents[DEFAULT_AGENT])
        return o, r, tm, tc, i

    def _get_reset_return(self, reset_info):
        """Add goal-dependent observation to the info dict."""
        o, i = super(MultiGoalIntersectionEnvBase, self)._get_reset_return(reset_info)

        if self.config["use_multigoal_intersection"]:
            for goal_name in self.engine.goal_manager.goals.keys():
                navi = self.engine.goal_manager.get_navigation(goal_name)
                goal_obs = self.observations["default_agent"].observe(self.agents[DEFAULT_AGENT], navi)
                i["obs/goals/{}".format(goal_name)] = goal_obs

        else:
            i["obs/goals/default"] = self.observations["default_agent"].observe(self.agents[DEFAULT_AGENT])

        return o, i

    def _reward_per_navigation(self, vehicle, navi, goal_name):
        """Compute the reward for the given goal. goal_name='default' means we use the vehicle's own navigation."""
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

        left, right = vehicle._dist_to_route_left_right(navigation=navi)
        out_of_route = (right < 0) or (left < 0)

        # Reward for speed, sign determined by whether in the correct lanes (instead of driving in the wrong
        # direction).
        reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) * positive_road
        if self._is_arrive_destination(vehicle):
            if self._is_arrive_destination(vehicle, goal_name):
                reward += self.config["success_reward"]
            else:
                # if goal_name == "default":
                #     print("WRONG WAY")
                reward = -self.config["wrong_way_penalty"]
        else:
            if self._is_out_of_road(vehicle):
                reward = -self.config["out_of_road_penalty"]
            elif vehicle.crash_vehicle:
                reward = -self.config["crash_vehicle_penalty"]
            elif vehicle.crash_object:
                reward = -self.config["crash_object_penalty"]
            elif vehicle.crash_sidewalk:
                reward = -self.config["crash_sidewalk_penalty"]
            elif out_of_route:
                # if goal_name == "default":
                #     print("OUT OF ROUTE")
                reward = -self.config["out_of_route_penalty"]

        return reward, navi.route_completion

    def reward_function(self, vehicle_id: str):
        """
        Compared to the original reward_function, we add goal-dependent reward to info dict.
        """
        vehicle = self.agents[vehicle_id]
        step_info = dict()

        # Compute goal-dependent reward and saved to step_info
        if self.config["use_multigoal_intersection"]:
            for goal_name in self.engine.goal_manager.goals.keys():
                navi = self.engine.goal_manager.get_navigation(goal_name)
                prefix = goal_name
                reward, route_completion = self._reward_per_navigation(vehicle, navi, goal_name)
                step_info[f"reward/goals/{prefix}"] = reward
                step_info[f"route_completion/goals/{prefix}"] = route_completion

        else:
            navi = vehicle.navigation
            goal_name = "default"
            reward, route_completion = self._reward_per_navigation(vehicle, navi, goal_name)
            step_info[f"reward/goals/{goal_name}"] = reward
            step_info[f"route_completion/goals/{goal_name}"] = route_completion

        default_reward, default_rc = self._reward_per_navigation(vehicle, vehicle.navigation, "default")
        step_info[f"reward/goals/default"] = default_reward
        step_info[f"route_completion/goals/default"] = default_rc
        step_info[f"reward/default_reward"] = default_reward
        step_info[f"route_completion"] = vehicle.navigation.route_completion

        return default_reward, step_info

    def _is_arrive_destination(self, vehicle, goal_name=None):
        """
        Compared to the original function, here we look up the navigation from goal_manager.

        Args:
            vehicle: The BaseVehicle instance.
            goal_name: The name of the goal. If None, return True if any goal is arrived.

        Returns:
            flag: Whether this vehicle arrives its destination.
        """

        if self.config["use_multigoal_intersection"]:
            if goal_name is None:
                ret = False
                for name in self.engine.goal_manager.goals.keys():
                    ret = ret or self._is_arrive_destination(vehicle, name)
                return ret

            if goal_name == "default":
                navi = self.agent.navigation
            else:
                navi = self.engine.goal_manager.get_navigation(goal_name)

        else:
            navi = vehicle.navigation

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
        done, done_info = super(MultiGoalIntersectionEnvBase, self).done_function(vehicle_id)
        vehicle = self.agents[vehicle_id]

        if self.config["use_multigoal_intersection"]:
            for goal_name in self.engine.goal_manager.goals.keys():
                done_info[f"arrive_dest/goals/{goal_name}"] = self._is_arrive_destination(vehicle, goal_name)

        else:
            done_info[f"arrive_dest/goals/default"] = done

        done_info["arrive_dest/goals/default"] = self._is_arrive_destination(vehicle, "default")

        return done, done_info


class MultiGoalIntersectionEnv(MultiGoalIntersectionEnvBase):
    current_goal = None

    @classmethod
    def default_config(cls):
        config = MultiGoalIntersectionEnvBase.default_config()
        config.update(
            {"goal_probabilities": {
                "u_turn": 0.25,
                "right_turn": 0.25,
                "go_straight": 0.25,
                "left_turn": 0.25,
            }}
        )
        return config

    def step(self, actions):
        o, r, tm, tc, i = super().step(actions)

        o = i['obs/goals/{}'.format(self.current_goal)]
        r = i['reward/goals/{}'.format(self.current_goal)]
        i['route_completion'] = i['route_completion/goals/{}'.format(self.current_goal)]
        i['arrive_dest'] = i['arrive_dest/goals/{}'.format(self.current_goal)]
        i['reward/goals/default'] = i['reward/goals/{}'.format(self.current_goal)]
        i['route_completion/goals/default'] = i['route_completion/goals/{}'.format(self.current_goal)]
        i['arrive_dest/goals/default'] = i['arrive_dest/goals/{}'.format(self.current_goal)]
        i["current_goal"] = self.current_goal
        return o, r, tm, tc, i

    def render(self, *args, **kwargs):
        if "text" in kwargs:
            kwargs["text"]["goal"] = self.current_goal
        else:
            kwargs["text"] = {"goal": self.current_goal}
        return super().render(*args, **kwargs)

    def reset(self, *args, **kwargs):
        o, i = super().reset(*args, **kwargs)

        # Sample a goal from the goal set
        if self.config["use_multigoal_intersection"]:
            p = self.config["goal_probabilities"]
            self.current_goal = np.random.choice(list(p.keys()), p=list(p.values()))

        else:
            self.current_goal = "default"

        o = i['obs/goals/{}'.format(self.current_goal)]
        i['route_completion'] = i['route_completion/goals/{}'.format(self.current_goal)]
        i['arrive_dest'] = i['arrive_dest/goals/{}'.format(self.current_goal)]
        i['reward/goals/default'] = i['reward/goals/{}'.format(self.current_goal)]
        i['route_completion/goals/default'] = i['route_completion/goals/{}'.format(self.current_goal)]
        i['arrive_dest/goals/default'] = i['arrive_dest/goals/{}'.format(self.current_goal)]
        i["current_goal"] = self.current_goal

        return o, i


if __name__ == "__main__":
    config = dict(
        use_render=True,
        manual_control=True,
        vehicle_config=dict(
            show_navi_mark=False,
            show_line_to_navi_mark=False,
            show_lidar=False,
            show_side_detector=True,
            show_lane_line_detector=True,
            show_line_to_dest=False,
            show_dest_mark=False,
        ),

        # ********************************************
        use_multigoal_intersection=True,
        # ********************************************

        # **{
        # "map_config": dict(
        #     lane_num=5,
        #     lane_width=3.5
        # ),
        # }
    )
    env = MultiGoalIntersectionEnv(config)
    episode_rewards = defaultdict(float)
    try:
        o, info = env.reset()

        # default_ckpt = env.vehicle.navigation.checkpoints[-1]
        # for goal, navi in env.engine.goal_manager.navigations.items():
        #     if navi.checkpoints[-1] == default_ckpt:
        #         break
        # assert np.all(o == info["obs/goals/{}".format(goal)])

        # goal = "default"
        goal = "left_turn"

        print('=======================')
        print("Full observation shape:\n\t", o.shape)
        print("Goal-agnostic observation shape:\n\t", NAVI_DIM + GOAL_DEPENDENT_STATE_DIM)
        print("Observation shape for each goals: ")
        for k in sorted(info.keys()):
            if k.startswith("obs/goals/"):
                print(f"\t{k}: {info[k].shape}")
        print('=======================')

        obs_recorder = defaultdict(list)

        s = 0
        for i in range(1, 1000000000):
            o, r, tm, tc, info = env.step([0, 1])

            # assert np.all(o == info["obs/goals/{}".format(goal)])
            # assert np.all(r == info["reward/goals/{}".format(goal)])

            done = tm or tc
            s += 1
            env.render()
            # env.render(mode="topdown")

            for k in info.keys():
                if k.startswith("obs/goals"):
                    obs_recorder[k].append(info[k])

            for k, v in info.items():
                if k.startswith("reward/goals"):
                    episode_rewards[k] += v

            if s % 20 == 0:
                print('\n===== timestep {} ====='.format(s))
                print('goal: ', goal)
                print('route completion:')
                for k in sorted(info.keys()):
                    if k.startswith("route_completion/goals/"):
                        print(f"\t{k}: {info[k]:.2f}")

                print('\nreward:')
                for k in sorted(info.keys()):
                    if k.startswith("reward/"):
                        print(f"\t{k}: {info[k]:.2f}")
                print('=======================')

            if done:
                print('\n===== timestep {} ====='.format(s))
                print("EPISODE DONE\n")
                print('route completion:')
                for k in sorted(info.keys()):
                    # kk = k.replace("/route_completion", "")
                    if k.startswith("route_completion/goals/"):
                        print(f"\t{k}: {info[k]:.2f}")

                print('\narrive destination (success):')
                for k in sorted(info.keys()):
                    # kk = k.replace("/arrive_dest", "")
                    if k.startswith("arrive_dest/goals/"):
                        print(f"\t{k}: {info[k]:.2f}")

                print('\nepisode_rewards:')
                for k in sorted(episode_rewards.keys()):
                    # kk = k.replace("/step_reward", "")
                    print(f"\t{k}: {episode_rewards[k]:.2f}")
                episode_rewards.clear()
                print('=======================')

            if done:
                # for t in range(i):
                #     # avg = [v[t] for k, v in obs_recorder.items()]
                #     v = np.stack([v[0] for k, v in obs_recorder.items()])

                print('\n\n\n')
                o, info = env.reset()

                default_ckpt = env.vehicle.navigation.checkpoints[-1]
                # for goal, navi in env.engine.goal_manager.navigations.items():
                #     if navi.checkpoints[-1] == default_ckpt:
                #         break
                #
                # assert np.all(o == info["obs/goals/{}".format(goal)])

                s = 0
    finally:
        env.close()
