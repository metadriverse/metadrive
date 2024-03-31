import copy
import os.path as osp

import numpy as np
from metadrive.utils import clip

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import TerminationState
from metadrive.manager.scenario_data_manager import ScenarioDataManager
from metadrive.envs.base_env import BaseEnv
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.obs.state_obs import StateObservation, NodeNetworkNavigation, np, gym, clip, norm, LidarStateObservation
from metadrive.component.navigation_module.trajectory_navigation import MultiAgentTrajectoryNavigation

from metadrive.constants import DEFAULT_AGENT

# TODO: Clean
MARL_SINGLE_WAYMO_ENV_CONFIG = {
    # "dataset_path": None,
    "num_agents": -1,  # Automatically determine how many agents
    # "start_case_index": 551,  # ===== Set the scene to 551 =====
    # "case_num": 1,
    # "waymo_env": True,
    "vehicle_config": {
        "agent_name": None,
        "spawn_lane_index": None,
        "navigation_module": MultiAgentTrajectoryNavigation
    },
    "no_static_traffic_vehicle": False,  # Allow static vehicles!
    "horizon": 200,  # The environment will end when environmental steps reach 200
    "delay_done": 0,  # If agent dies, it will be removed immediately from the scene.
    # "sequential_seed": False,
    # "save_memory": False,
    # "save_memory_max_len": 50,

    "store_map": True,
    "store_map_buffer_size": 10,

    # ===== New config ===
    "randomized_dynamics": None,  #####

    "discrete_action_dim": 5,

    "relax_out_of_road_done": False,

    "replay_traffic_vehicle": False,

    # ===== Set scene to [0, 1000] =====
    "start_case_index": 0,
    "case_num": 1000,
    # "dataset_path": WAYMO_DATASET_PATH,

    "distance_penalty": 0

}
WAYMO_DATASET_PATH = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), "dataset", "env_num_1165_waymo")
MARL_WAYMO_ENV_CONFIG = copy.deepcopy(MARL_SINGLE_WAYMO_ENV_CONFIG)
MARL_WAYMO_ENV_CONFIG.update(
    {

    }
)


class NewStateObservation(StateObservation):
    ego_state_obs_dim = 11

    def vehicle_state(self, vehicle):
        """
        Wrap vehicle states to list
        """
        # update out of road
        info = []

        # Change: Always add these two
        # The length/width of the target vehicle
        info.append(clip(vehicle.LENGTH / 20, 0.0, 1.0))
        info.append(clip(vehicle.WIDTH / 20, 0.0, 1.0))

        if hasattr(vehicle, "side_detector") and vehicle.side_detector.available:
            # If side detector (a Lidar scanning road borders) is turn on, then add the cloud points of side detector
            info += vehicle.side_detector.perceive(vehicle, vehicle.engine.physics_world.static_world).cloud_points

        # Change: Always add these two. vehicle.navigation don't have map anymore.
        # Randomly pick a large number as total width instead!

        # If the side detector is turn off, then add the distance to left and right road borders as state.
        lateral_to_left, lateral_to_right, = vehicle.dist_to_left_side, vehicle.dist_to_right_side

        # total_width = float((vehicle.navigation.map.MAX_LANE_NUM + 1) * vehicle.navigation.map.MAX_LANE_WIDTH)
        total_width = 50
        lateral_to_left /= total_width
        lateral_to_right /= total_width
        info += [clip(lateral_to_left, 0.0, 1.0), clip(lateral_to_right, 0.0, 1.0)]

        if vehicle.navigation is None or vehicle.navigation.current_ref_lanes is None or \
                vehicle.navigation.current_ref_lanes[-1] is None:
            info += [0] * 5
        else:
            current_reference_lane = vehicle.navigation.current_ref_lanes[-1]
            info += [

                # The angular difference between vehicle's heading and the lane heading at this location.
                vehicle.heading_diff(current_reference_lane),

                # The velocity of target vehicle
                clip((vehicle.speed + 1) / (vehicle.max_speed + 1), 0.0, 1.0),

                # Acceleration
                clip((vehicle.speed - vehicle.last_speed) / 10 + 0.5, 0.0, 1.0),
                clip((vehicle.velocity[0] - vehicle.last_velocity[0]) / 10 + 0.5, 0.0, 1.0),
                clip((vehicle.velocity[1] - vehicle.last_velocity[1]) / 10 + 0.5, 0.0, 1.0),

                # Current steering
                clip((vehicle.steering / vehicle.MAX_STEERING + 1) / 2, 0.0, 1.0),

                # Change: Remove last action. This cause issue when collecting expert data since expert data do not
                # have any action!!!!
                # The normalized actions at last steps
                # clip((vehicle.last_current_action[0][0] + 1) / 2, 0.0, 1.0),
                # clip((vehicle.last_current_action[0][1] + 1) / 2, 0.0, 1.0)
            ]

        # Current angular acceleration (yaw rate)
        heading_dir_last = vehicle.last_heading_dir
        heading_dir_now = vehicle.heading
        cos_beta = heading_dir_now.dot(heading_dir_last) / (norm(*heading_dir_now) * norm(*heading_dir_last))
        beta_diff = np.arccos(clip(cos_beta, 0.0, 1.0))
        yaw_rate = beta_diff / 0.1
        info.append(clip(yaw_rate, 0.0, 1.0))

        if vehicle.lane_line_detector.available:

            # If lane line detector (a Lidar scanning current lane borders) is turn on,
            # then add the cloud points of lane line detector
            info += vehicle.lane_line_detector.perceive(vehicle, vehicle.engine.physics_world.static_world).cloud_points

        else:

            # If the lane line detector is turn off, then add the offset of current position
            # against the central of current lane to the state. If vehicle is centered in the lane, then the offset
            # is 0 and vice versa.
            _, lateral = vehicle.lane.local_coordinates(vehicle.position)
            info.append(clip((lateral * 2 / vehicle.navigation.map.MAX_LANE_WIDTH + 1.0) / 2.0, 0.0, 1.0))

        return info


class NewLidarStateObservation(LidarStateObservation):
    def __init__(self, vehicle_config):
        super(NewLidarStateObservation, self).__init__(vehicle_config)
        self.state_obs = NewStateObservation(vehicle_config)



class NewWaymoObservation(NewLidarStateObservation):
    MAX_LATERAL_DIST = 20

    def __init__(self, *args, **kwargs):
        super(NewWaymoObservation, self).__init__(*args, **kwargs)
        self.lateral_dist = 0

    @property
    def observation_space(self):
        shape = list(self.state_obs.observation_space.shape)
        if self.config["lidar"]["num_lasers"] > 0 and self.config["lidar"]["distance"] > 0:
            # Number of lidar rays and distance should be positive!
            lidar_dim = self.config["lidar"]["num_lasers"] + self.config["lidar"]["num_others"] * 4
            if self.config["lidar"]["add_others_navi"]:
                lidar_dim += self.config["lidar"]["num_others"] * 4
            shape[0] += lidar_dim
        shape[0] += 1  # add one dim for sensing lateral distance to the sdc trajectory
        return gym.spaces.Box(-0.0, 1.0, shape=tuple(shape), dtype=np.float32)

    def state_observe(self, vehicle):
        ret = super(NewWaymoObservation, self).state_observe(vehicle)
        lateral_obs = self.lateral_dist / self.MAX_LATERAL_DIST
        return np.concatenate([ret, [clip((lateral_obs + 1) / 2, 0.0, 1.0)]])

    def reset(self, env, vehicle=None):
        super(NewWaymoObservation, self).reset(env, vehicle)
        self.lateral_dist = 0





class MARLSingleWaymoEnv(ScenarioEnv, MultiAgentMetaDrive):
    @classmethod
    def default_config(cls):

        config = super(MARLSingleWaymoEnv, cls).default_config()
        config.update(MARL_SINGLE_WAYMO_ENV_CONFIG)
        # config.register_type("randomized_dynamics", None, str)
        return config

    def setup_engine(self):
        self.in_stop = False

        # Call the setup_engine of BaseEnv
        self.engine.accept("r", self.reset)
        self.engine.accept("p", self.capture)

        # TODO: Do we still need these two?
        self.engine.register_manager("data_manager", WaymoDataManager())
        self.engine.register_manager("map_manager", MAWaymoMapManager())


        # TODO: Do we still need this?
        # Overwrite the Agent Manager
        self.agent_manager = MAWaymoAgentManager(
            init_observations=self._get_observations(),
            init_action_space=self._get_action_space(),
            store_map=self.config["store_map"],
            store_map_buffer_size=self.config["store_map_buffer_size"]
        )
        self.engine.register_manager("agent_manager", self.agent_manager)


        self.engine.accept("p", self.stop)
        self.engine.accept("q", self.switch_to_third_person_view)
        self.engine.accept("b", self.switch_to_top_down_view)

    def __init__(self, config):
        # data_path = config.get("dataset_path", self.default_config()["dataset_path"])
        # assert osp.exists(data_path), \
        #     "Can not find dataset: {}, Please download it from: " \
        #     "https://github.com/metadriverse/metadrive-scenario/releases.".format(data_path)

        config = copy.deepcopy(config)

        # if config.get("discrete_action"):
        #     self._waymo_discrete_action = True
        #     config["discrete_action"] = False
        # else:
        #     self._waymo_discrete_action = False

        # config["waymo_data_directory"] = data_path
        super(MARLSingleWaymoEnv, self).__init__(config)
        self.agent_steps = 0
        self._sequential_seed = None

        # self.dynamics_parameters_mean = None
        # self.dynamics_parameters_std = None
        # self.dynamics_function = None

    # def set_dynamics_parameters_distribution(self, dynamics_parameters_mean=None, dynamics_parameters_std=None,
    #                                          dynamics_function=None):
    #     if dynamics_parameters_mean is not None:
    #         assert isinstance(dynamics_parameters_mean, np.ndarray)
    #     self.dynamics_parameters_mean = dynamics_parameters_mean
    #     self.dynamics_parameters_std = dynamics_parameters_std
    #     self.dynamics_function = dynamics_function

    def reset(self, *args, **kwargs):

        force_seed = None

        # if self.config["randomized_dynamics"] == "naive":
        #     def _d(environment_seed=None, agent_name=None, latent_dict=None):
        #         s = np.random.randint(3)
        #         if s == 0:
        #             return np.random.normal(-0.5, 0.2, size=5), {"mode": s}
        #         elif s == 1:
        #             return np.random.normal(0, 0.2, size=5), {"mode": s}
        #         elif s == 2:
        #             return np.random.normal(0.5, 0.2, size=5), {"mode": s}
        #         else:
        #             raise ValueError()
        #
        #     self.dynamics_function = _d

        finish = False

        while not finish:

            if self.config["sequential_seed"] and "force_seed" in kwargs:
                self._sequential_seed = kwargs["force_seed"]

            if not self.config["sequential_seed"] and "force_seed" in kwargs:
                force_seed = kwargs["force_seed"]

            if self.config["sequential_seed"]:
                if self._sequential_seed is None:
                    self._sequential_seed = self.config["start_case_index"]
                force_seed = self._sequential_seed
                self._sequential_seed += 1
                if self._sequential_seed >= self.config["start_case_index"] + self.config["case_num"]:
                    self._sequential_seed = self.config["start_case_index"]

            # if self.config["randomized_dynamics"]:
            #     if not isinstance(self.agent_manager, MAWaymoAgentManager):
            #         ret = super(MARLSingleWaymoEnv, self).reset(*args, **kwargs)
            #
            #     assert isinstance(self.agent_manager, MAWaymoAgentManager)
            #
            #     # For some reasons, env.reset could be called when the agent_manager is not set to
            #     # WaymoAgentManager yet.
            #     self.agent_manager.set_dynamics_parameters_distribution(
            #         dynamics_parameters_mean=self.dynamics_parameters_mean,
            #         dynamics_parameters_std=self.dynamics_parameters_std,
            #         dynamics_function=self.dynamics_function,
            #         latent_dict=self.latent_dict if hasattr(self, "latent_dict") else None
            #     )

            if "force_seed" in kwargs:
                kwargs.pop("force_seed")

            # TODO: What's the new seed?
            ret = super(MARLSingleWaymoEnv, self).reset(*args, force_seed=force_seed, **kwargs)

            # Since we are using Waymo real data, it is possible that the vehicle crashes with solid line already.
            # Remove those vehicles since it will terminate very soon during RL interaction.
            for agent_name, vehicle in self.agent_manager.active_agents.items():
                done = self._is_out_of_road(vehicle)
                if done:
                    self.agent_manager.put_to_static_list(vehicle)
                    if agent_name in ret:
                        ret.pop(agent_name)

            finish = len(self.vehicles) > 0

        self.agent_steps = 0
        # self.dynamics_parameters_recorder = dict()
        return ret

    # def set_dynamics_parameters(self, mean, std):
    #     assert self.config["lcf_dist"] == "normal"
    #     self.current_lcf_mean = mean
    #     self.current_lcf_std = std
    #     assert std > 0.0
    #     assert -1.0 <= self.current_lcf_mean <= 1.0

    def step(self, actions):

        # Let WaymoEnv to process action
        # This is equivalent to call MetaDriveEnv to process actions (it can deal with MA input!)
        obses, rewards, dones, infos = BaseEnv.step(self, actions)

        # Process agents according to whether they are done

        # def _after_vehicle_done(self, obs=None, reward=None, dones: dict = None, info=None):
        for v_id, v_info in infos.items():
            infos[v_id][TerminationState.MAX_STEP] = False
            if v_info.get("episode_length", 0) >= self.config["horizon"]:
                if dones[v_id] is not None:
                    infos[v_id][TerminationState.MAX_STEP] = True
                    dones[v_id] = True
                    self.dones[v_id] = True

            # Process a special case where RC > 1.
            # Even though we don't know what causes it, we should at least workaround it.
            if "route_completion" in infos[v_id]:
                rc = infos[v_id]["route_completion"]
                if (rc > 1.0 or rc < -0.1) and dones[v_id] is not None:
                    if rc > 1.0:
                        infos[v_id][TerminationState.SUCCESS] = True
                    dones[v_id] = True
                    self.dones[v_id] = True

        for dead_vehicle_id, done in dones.items():
            if done:
                self.agent_manager.finish(
                    dead_vehicle_id, ignore_delay_done=infos[dead_vehicle_id].get(TerminationState.SUCCESS, False)
                )
                self._update_camera_after_finish()
            # return obs, reward, dones, info

        # PZH: Do not respawn new vehicle!
        # Update respawn manager
        # if self.episode_step >= self.config["horizon"]:
        #     self.agent_manager.set_allow_respawn(False)
        # new_obs_dict, new_info_dict = self._respawn_vehicles(randomize_position=self.config["random_traffic"])
        # if new_obs_dict:
        #     for new_id, new_obs in new_obs_dict.items():
        #         o[new_id] = new_obs
        #         r[new_id] = 0.0
        #         i[new_id] = new_info_dict[new_id]
        #         d[new_id] = False

        self.agent_steps += len(self.vehicles)

        # Update __all__
        d_all = False
        if self.config["horizon"] is not None:  # No agent alive or a too long episode happens
            if self.episode_step >= self.config["horizon"]:
                d_all = True
        if len(self.vehicles) == 0:  # No agent alive
            d_all = True
        dones["__all__"] = d_all
        if dones["__all__"]:
            for k in dones.keys():
                dones[k] = True

        for k in infos.keys():
            infos[k]["agent_steps"] = self.agent_steps
            infos[k]["environment_seed"] = self.engine.global_seed
            infos[k]["vehicle_id"] = k

            # if dones[k]:
            #     infos[k]["raw_state"] = None
            #     infos[k]["dynamics"] = self.dynamics_parameters_recorder[k]
            # else:
            try:
                v = self.agent_manager.get_agent(k)
                # infos[k]["raw_state"] = v.get_raw_state()
                infos[k]["dynamics"] = v.get_dynamics_parameters()
                self.dynamics_parameters_recorder[k] = infos[k]["dynamics"]

            except (ValueError, KeyError):
                # infos[k]["raw_state"] = None
                if k in self.dynamics_parameters_recorder:
                    infos[k]["dynamics"] = self.dynamics_parameters_recorder[k]
                else:
                    infos[k]["dynamics"] = None

        return obses, rewards, dones, infos

    # TODO: Clean
    # def _preprocess_actions(self, action):
    #     if self._waymo_discrete_action:
    #         new_action = {}
    #         discrete_action_dim = self.config["discrete_action_dim"]
    #         for k, v in action.items():
    #             assert 0 <= v < discrete_action_dim * discrete_action_dim
    #             a0 = v % discrete_action_dim
    #             a0 = _get_action(a0, discrete_action_dim)
    #             a1 = v // discrete_action_dim
    #             a1 = _get_action(a1, discrete_action_dim)
    #             new_action[k] = [a0, a1]
    #         action = new_action
    #
    #     return super(MARLSingleWaymoEnv, self)._preprocess_actions(action)

    def _get_action_space(self):
        # TODO: Clean
        # if self._waymo_discrete_action:
        #     from gym.spaces import Discrete
        #     discrete_action_dim = self.config["discrete_action_dim"]
        #     return {DEFAULT_AGENT: Discrete(discrete_action_dim * discrete_action_dim)}

        return {
            DEFAULT_AGENT: BaseVehicle.get_action_space_before_init(
                self.config["vehicle_config"]["extra_action_dim"], self.config["discrete_action"],
                self.config["discrete_steering_dim"], self.config["discrete_throttle_dim"]
            )
        }

    def _get_observations(self):
        return {DEFAULT_AGENT: self.get_single_observation(self.config["vehicle_config"])}

    def get_single_observation(self, vehicle_config):
        # from newcopo.metadrive_scenario.marl_envs.observation import NewWaymoObservation
        return NewWaymoObservation(vehicle_config)

    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        raise ValueError("check marl_waymo_env.py")
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # update obs
        self.observations[vehicle_id].lateral_dist = \
            self.engine.map_manager.current_routes[vehicle_id].local_coordinates(vehicle.position)[-1]

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral_reward"]:
            lateral_factor = clip(1 - 2 * abs(lateral_now) / 6, 0.0, 1.0)
        else:
            lateral_factor = 1.0

        reward = 0
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor

        step_info["step_reward"] = reward

        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]

        step_info["track_length"] = vehicle.navigation.reference_trajectory.length
        step_info["current_distance"] = vehicle.navigation.reference_trajectory.local_coordinates(vehicle.position)[0]
        rc = step_info["current_distance"] / step_info["track_length"]
        step_info["route_completion"] = rc

        # print("Vehicle {} Track Length {:.3f} Current Dis {:.3f} Route Completion {:.4f}".format(
        #     vehicle, step_info["track_length"] , step_info["current_distance"] , step_info["route_completion"]
        # ))

        return reward, step_info

    # def _is_arrive_destination(self, vehicle):
    #     if np.linalg.norm(vehicle.position - self.engine.map_manager.dest_points[vehicle.name]) < 5:
    #         return True
    #     else:
    #         return False

    def _is_arrive_destination(self, vehicle):
        long, lat = vehicle.navigation.reference_trajectory.local_coordinates(vehicle.position)

        total_length = vehicle.navigation.reference_trajectory.length
        current_distance = long

        # agent_name = self.agent_manager.object_to_agent(vehicle.name)
        # threshold = 5

        # if np.linalg.norm(vehicle.position - self.engine.map_manager.dest_points[agent_name]) < threshold:
        #     return True
        # elif current_distance + threshold > total_length:  # Route Completion ~= 1.0
        #     return True
        # else:
        #     return False

        # Update 2022-02-05: Use RC as the only criterion to determine arrival.
        route_completion = current_distance / total_length
        if route_completion > 0.95:  # Route Completion ~= 1.0
            return True
        else:
            return False

    def _is_out_of_road(self, vehicle):
        # A specified function to determine whether this vehicle should be done.

        if self.config["relax_out_of_road_done"]:
            agent_name = self.agent_manager.object_to_agent(vehicle.name)
            lat = abs(self.observations[agent_name].lateral_dist)
            done = lat > 10
            done = done or vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line
            return done

        done = vehicle.crash_sidewalk or vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line
        if self.config["out_of_route_done"]:
            agent_name = self.agent_manager.object_to_agent(vehicle.name)
            done = done or abs(self.observations[agent_name].lateral_dist) > 10
        return done




if __name__ == "__main__":

    from tqdm import tqdm
    from metadrive.engine.asset_loader import AssetLoader

    # parser = argparse.ArgumentParser()
    # # parser.add_argument("--manual_control", action="store_true")
    # # parser.add_argument("--scenario_start", type=int, required=True)
    # # parser.add_argument("--scenario_end", type=int, required=True)
    # parser.add_argument("--topdown", action="store_true")
    # # parser.add_argument("--dataset", required=True, type=str)
    # parser.add_argument("--idm_traffic", action="store_true")
    # args = parser.parse_args()
    # is_waymo = "waymo" in args.dataset

    # ====================
    env_config = {"use_render": False}

    # SUMMARY:

    # If you want the traffic flow to be log-replay, set it to True:
    # env_config["replay"] = False
    # env_config["replay"] = True

    # Manual control
    env_config["manual_control"] = False

    # video_name = "StandardWaymoEnv-TrafficReplay-EgoIDM.mp4"
    # video_name = "MAWaymoEnv.mp4"

    # If you want to use RL agent to control the agent, comment out this:
    # from metadrive.policy.replay_policy import ReplayEgoCarPolicy

    # env_config["agent_policy"] = ReplayEgoCarPolicy

    # from metadrive.policy.idm_policy import WaymoIDMPolicy, EgoWaymoIDMPolicy
    # env_config["agent_policy"] = EgoWaymoIDMPolicy
    # env_config["agent_policy"] = WaymoIDMPolicy

    # ====================

    # dataset_path = STANDARD_WAYMO_ENV_PATH
    extra_env_config = env_config
    extra_env_config = extra_env_config or {}

    # print("Loading data from: ", dataset_path)

    config = dict(
        # dataset_path=dataset_path,
        # scenario_start=scenario_start,
        # scenario_end=scenario_end,
        # seed=0,
        # waymo_env=True,
        # extra_env_config=extra_env_config,
        # random_set_seed=True,

        # horizon=20,
        **extra_env_config
    )

    # color = sns.color_palette("colorblind")[2]
    # tmp_folder = video_name + ".TMP"
    # os.makedirs(tmp_folder, exist_ok=True)

    env = MARLSingleWaymoEnv(config)

    env.reset()

    # length = env._env.vehicle.navigation.reference_trajectory.length

    # env.vehicle._panda_color = color
    for _ in tqdm(range(1), desc="Episode"):
        for t in tqdm(range(1000), desc="Step"):

            # for step in range(1000):
            o, r, d, i = env.step({key: [-1, 1] for key in env.vehicles.keys()})

            # print("Number of agents: ", len(env.vehicles), len(env.action_space.spaces),
            #       len(env.observation_space.spaces))

            # Render for poping window:
            # ret = env.render(mode="top_down", film_size=(2000, 2000), screen_size=(2000, 2000))

            # Render for saving video:

            # Enable this line so no window will pop up
            # os.environ['SDL_VIDEODRIVER'] = 'dummy'
            # import pygame

            # ret = env.render(mode="top_down", film_size=(3000, 3000), screen_size=(1920, 1080),
            #                  track_target_vehicle=True)
            ret = env.render(
                mode="top_down", film_size=(3000, 3000), screen_size=(3000, 3000), track_target_vehicle=False
            )
            # pygame.image.save(ret, "{}/{}.png".format(tmp_folder, t))

            # if env.episode_step >= len(env.vehicle.navigation.reference_trajectory.segment_property):
            # if d or env.episode_step >= len(env.vehicle.navigation.reference_trajectory.segment_property):
            if d["__all__"]:
                print("Done!")
                env.reset()
                # length = env.vehicle.navigation.reference_trajectory.length
                break

    # image_to_video(video_name, tmp_folder)
    # shutil.rmtree(tmp_folder)
