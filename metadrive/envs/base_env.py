import logging
import time
from collections import defaultdict
from typing import Union, Dict, AnyStr, Optional, Tuple, Callable

import gymnasium as gym
import numpy as np
from panda3d.core import PNMImage

from metadrive import constants
from metadrive.component.sensors.base_camera import BaseCamera
from metadrive.component.sensors.dashboard import DashBoard
from metadrive.component.sensors.distance_detector import LaneLineDetector, SideDetector
from metadrive.component.sensors.lidar import Lidar
from metadrive.constants import RENDER_MODE_NONE, DEFAULT_AGENT
from metadrive.constants import RENDER_MODE_ONSCREEN, RENDER_MODE_OFFSCREEN
from metadrive.constants import TerminationState, TerrainProperty
from metadrive.engine.engine_utils import initialize_engine, close_engine, \
    engine_initialized, set_global_random_seed, initialize_global_config, get_global_config
from metadrive.engine.logger import get_logger, set_log_level
from metadrive.manager.agent_manager import VehicleAgentManager
from metadrive.manager.record_manager import RecordManager
from metadrive.manager.replay_manager import ReplayManager
from metadrive.obs.image_obs import ImageStateObservation
from metadrive.obs.observation_base import BaseObservation
from metadrive.obs.observation_base import DummyObservation
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.scenario.utils import convert_recorded_scenario_exported
from metadrive.utils import Config, merge_dicts, get_np_random, concat_step_infos
from metadrive.version import VERSION
from metadrive.component.navigation_module.base_navigation import BaseNavigation

from pynput import keyboard

BASE_DEFAULT_CONFIG = dict(

    # ===== agent =====
    random_agent_model=False,
    agent_configs={DEFAULT_AGENT: dict(use_special_color=True, spawn_lane_index=None)},

    # ===== multi-agent =====
    num_agents=1,
    is_multi_agent=False,
    allow_respawn=False,
    delay_done=0,

    # ===== Action/Control =====
    agent_policy=EnvInputPolicy,
    manual_control=False,
    controller="keyboard",
    discrete_action=False,
    use_multi_discrete=False,
    discrete_steering_dim=5,
    discrete_throttle_dim=5,
    action_check=False,

    # ===== Observation =====
    norm_pixel=True,
    stack_size=3,
    image_observation=False,
    agent_observation=None,

    # ===== Termination =====
    horizon=None,
    truncate_as_terminate=False,

    # ===== Main Camera =====
    use_chase_camera_follow_lane=False,
    camera_height=2.2,
    camera_dist=7.5,
    camera_pitch=None,  # degree
    camera_smooth=True,
    camera_smooth_buffer_size=20,
    camera_fov=65,
    prefer_track_agent=None,
    top_down_camera_initial_x=0,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=200,

    # ===== Vehicle =====
    vehicle_config=dict(
        vehicle_model="default",
        enable_reverse=False,
        show_navi_mark=True,
        show_dest_mark=False,
        show_line_to_dest=False,
        show_line_to_navi_mark=False,
        use_special_color=False,
        no_wheel_friction=False,
        image_source="rgb_camera",
        navigation_module=None,
        spawn_lane_index=None,
        destination=None,
        spawn_longitude=5.0,
        spawn_lateral=0.0,
        spawn_position_heading=None,
        spawn_velocity=None,  # m/s
        spawn_velocity_car_frame=False,
        overtake_stat=False,
        random_color=False,
        width=None,
        length=None,
        height=None,
        mass=None,
        top_down_width=None,
        top_down_length=None,
        lidar=dict(
            num_lasers=240, distance=50, num_others=0, gaussian_noise=0.0, dropout_prob=0.0, add_others_navi=False
        ),
        side_detector=dict(num_lasers=0, distance=50, gaussian_noise=0.0, dropout_prob=0.0),
        lane_line_detector=dict(num_lasers=0, distance=20, gaussian_noise=0.0, dropout_prob=0.0),
        show_lidar=False,
        show_side_detector=False,
        show_lane_line_detector=False,
        light=False,
    ),

    # ===== Sensors =====
    sensors=dict(lidar=(Lidar, ), side_detector=(SideDetector, ), lane_line_detector=(LaneLineDetector, )),

    # ===== Engine Core config =====
    use_render=False,
    window_size=(1200, 900),
    physics_world_step_size=2e-2,
    decision_repeat=5,
    image_on_cuda=False,
    _render_mode=RENDER_MODE_NONE,
    force_render_fps=None,
    force_destroy=False,
    num_buffering_objects=200,
    render_pipeline=False,
    daytime="19:00",  # use string like "13:40", We usually set this by editor in toolkit
    shadow_range=50,
    multi_thread_render=True,
    multi_thread_render_mode="Cull",
    preload_models=True,
    disable_model_compression=True,

    # ===== Terrain =====
    map_region_size=1024,
    cull_lanes_outside_map=False,
    drivable_area_extension=7,
    height_scale=50,
    use_mesh_terrain=False,
    full_size_mesh=True,
    show_crosswalk=True,
    show_sidewalk=True,

    # ===== Debug =====
    pstats=False,
    debug=False,
    debug_panda3d=False,
    debug_physics_world=False,
    debug_static_world=False,
    log_level=logging.INFO,
    show_coordinates=False,

    # ===== GUI =====
    show_fps=True,
    show_logo=True,
    show_mouse=True,
    show_skybox=True,
    show_terrain=True,
    show_interface=True,
    show_policy_mark=False,
    show_interface_navi_mark=True,
    interface_panel=["dashboard"],

    # ===== Record/Replay Metadata =====
    record_episode=False,
    replay_episode=None,
    only_reset_when_replay=False,
    force_reuse_object_name=False,

    # ===== randomization =====
    num_scenarios=1  # the number of scenarios in this environment
)


class BaseEnv(gym.Env):
    _DEBUG_RANDOM_SEED: Union[int, None] = None

    @classmethod
    def default_config(cls) -> Config:
        return Config(BASE_DEFAULT_CONFIG)

    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        self.logger = get_logger()
        set_log_level(config.get("log_level", logging.DEBUG if config.get("debug", False) else logging.INFO))
        merged_config = self.default_config().update(config)
        global_config = self._post_process_config(merged_config)

        self.config = global_config
        initialize_global_config(self.config)

        self.num_agents = self.config["num_agents"]
        self.is_multi_agent = self.config["is_multi_agent"]
        if not self.is_multi_agent:
            assert self.num_agents == 1
        else:
            assert not self.config["image_on_cuda"], "Image on cuda don't support Multi-agent!"
        assert isinstance(self.num_agents, int) and (self.num_agents > 0 or self.num_agents == -1)

        self.agent_manager = self._get_agent_manager()

        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)

        self.in_stop = False

        self.start_index = 0
        self.num_scenarios = self.config["num_scenarios"]

    def _post_process_config(self, config):
        self.logger.info("Environment: {}".format(self.__class__.__name__))
        self.logger.info("MetaDrive version: {}".format(VERSION))
        if not config["show_interface"]:
            config["interface_panel"] = []

        n = config["map_region_size"]
        assert (n & (n - 1)) == 0 and 0 < n <= 2048, "map_region_size should be pow of 2 and < 2048."
        TerrainProperty.map_region_size = config["map_region_size"]

        if not config["use_render"] and not config["image_observation"]:
            filtered = {}
            for id, cfg in config["sensors"].items():
                if len(cfg) > 0 and not issubclass(cfg[0], BaseCamera) and id != "main_camera":
                    filtered[id] = cfg
            config["sensors"] = filtered
            config["interface_panel"] = []

        if config["use_render"] or "main_camera" in config["sensors"]:
            config["sensors"]["main_camera"] = ("MainCamera", *config["window_size"])

        to_use = []
        if not config["render_pipeline"] and config["show_interface"] and "main_camera" in config["sensors"]:
            for panel in config["interface_panel"]:
                if panel == "dashboard":
                    config["sensors"]["dashboard"] = (DashBoard, )
                if panel not in config["sensors"]:
                    self.logger.warning(
                        "Fail to add sensor: {} to the interface. Remove it from panel list!".format(panel)
                    )
                elif panel == "main_camera":
                    self.logger.warning("main_camera can not be added to interface_panel, remove")
                else:
                    to_use.append(panel)
        config["interface_panel"] = to_use

        sensor_cfg = self.default_config()["sensors"].update(config["sensors"])
        config["sensors"] = sensor_cfg

        sensors_str = ""
        for _id, cfg in config["sensors"].items():
            sensors_str += "{}: {}{}, ".format(_id, cfg[0] if isinstance(cfg[0], str) else cfg[0].__name__, cfg[1:])
        self.logger.info("Sensors: [{}]".format(sensors_str[:-2]))

        if config["use_render"]:
            assert "main_camera" in config["sensors"]
            config["_render_mode"] = RENDER_MODE_ONSCREEN
        else:
            config["_render_mode"] = RENDER_MODE_NONE
            for sensor in config["sensors"].values():
                if sensor[0] == "MainCamera" or (issubclass(sensor[0], BaseCamera) and sensor[0] != DashBoard):
                    config["_render_mode"] = RENDER_MODE_OFFSCREEN
                    break
        self.logger.info("Render Mode: {}".format(config["_render_mode"]))
        self.logger.info("Horizon (Max steps per agent): {}".format(config["horizon"]))
        if config["truncate_as_terminate"]:
            self.logger.warning(
                "When reaching max steps, both 'terminate' and 'truncate will be True."
                "Generally, only the `truncate` should be `True`."
            )
        return config

    def _get_observations(self) -> Dict[str, "BaseObservation"]:
        return {DEFAULT_AGENT: self.get_single_observation()}

    def _get_agent_manager(self):
        return VehicleAgentManager(init_observations=self._get_observations())

    def lazy_init(self):
        if engine_initialized():
            return
        initialize_engine(self.config)
        self.setup_engine()
        if self.config["vehicle_config"]["navigation_module"] is None:
            self.config["vehicle_config"]["navigation_module"] = BaseNavigation(
                vehicle_config=self.config["vehicle_config"]
            )
        self._after_lazy_init()
        self.logger.info(
            "Start Scenario Index: {}, Num Scenarios : {}".format(
                self.engine.gets_start_index(self.config), self.config.get("num_scenarios", 1)
            )
        )

    @property
    def engine(self):
        from metadrive.engine.engine_utils import get_engine
        return get_engine()

    def _after_lazy_init(self):
        pass

    def step(self, actions: Union[Union[np.ndarray, list], Dict[AnyStr, Union[list, np.ndarray]], int]):
        actions = self._preprocess_actions(actions)  # preprocess environment input
        engine_info = self._step_simulator(actions)  # step the simulation
        while self.in_stop:
            self.engine.taskMgr.step()  # pause simulation
        return self._get_step_return(actions, engine_info=engine_info)  # collect observation, reward, termination
    def _preprocess_actions(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray], int]) \
            -> Union[np.ndarray, Dict[AnyStr, np.ndarray], int]:
        if not self.is_multi_agent:
            actions = {v_id: actions for v_id in self.agents.keys()}
        else:
            if self.config["action_check"]:
                given_keys = set(actions.keys())
                have_keys = set(self.agents.keys())
                assert given_keys == have_keys, "The input actions: {} have incompatible keys with existing {}!".format(
                    given_keys, have_keys
                )
            else:
                actions = {v_id: actions[v_id] for v_id in self.agents.keys()}
        return actions

    def _step_simulator(self, actions):
        scene_manager_before_step_infos = self.engine.before_step(actions)
        self.engine.step(self.config["decision_repeat"])
        scene_manager_after_step_infos = self.engine.after_step()

        return merge_dicts(
            scene_manager_after_step_infos, scene_manager_before_step_infos, allow_new_keys=True, without_copy=True
        )

    def reward_function(self, object_id: str) -> Tuple[float, Dict]:
        self.logger.warning("Reward function is not implemented. Return reward = 0", extra={"log_once": True})
        return 0, {}

    def cost_function(self, object_id: str) -> Tuple[float, Dict]:
        self.logger.warning("Cost function is not implemented. Return cost = 0", extra={"log_once": True})
        return 0, {}

    def done_function(self, object_id: str) -> Tuple[bool, Dict]:
        self.logger.warning("Done function is not implemented. Return Done = False", extra={"log_once": True})
        return False, {}

    def render(self, text: Optional[Union[dict, str]] = None, mode=None, *args, **kwargs) -> Optional[np.ndarray]:
        if mode in ["top_down", "topdown", "bev", "birdview"]:
            ret = self._render_topdown(text=text, *args, **kwargs)
            return ret
        if self.config["use_render"] or self.engine.mode != RENDER_MODE_NONE:
            self.engine.render_frame(text)
        else:
            self.logger.warning(
                "Panda Rendering is off now, can not render. Please set config['use_render'] = True!",
                exc_info={"log_once": True}
            )
        return None

    def reset(self, seed: Union[None, int] = None):
        if self.logger is None:
            self.logger = get_logger()
            log_level = self.config.get("log_level", logging.DEBUG if self.config.get("debug", False) else logging.INFO)
            set_log_level(log_level)
        self.lazy_init()
        self._reset_global_seed(seed)
        if self.engine is None:
            raise ValueError(
                "Current MetaDrive instance is broken. Please make sure there is only one active MetaDrive "
                "environment exists in one process. You can try to call env.close() and then call "
                "env.reset() to rescue this environment. However, a better and safer solution is to check the "
                "singleton of MetaDrive and restart your program."
            )
        reset_info = self.engine.reset()
        self.reset_sensors()
        self.engine.taskMgr.step()
        if self.top_down_renderer is not None:
            self.top_down_renderer.clear()
            self.engine.top_down_renderer = None

        self.dones = {agent_id: False for agent_id in self.agents.keys()}
        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)

        assert (len(self.agents) == self.num_agents) or (self.num_agents == -1), \
            "Agents: {} != Num_agents: {}".format(len(self.agents), self.num_agents)
        assert self.config is self.engine.global_config is get_global_config(), "Inconsistent config may bring errors!"
        return self._get_reset_return(reset_info)

    def reset_sensors(self):
        if self.main_camera is not None:
            self.main_camera.reset()
            if hasattr(self, "agent_manager"):
                bev_cam = self.main_camera.is_bird_view_camera() and self.main_camera.current_track_agent is not None
                agents = list(self.engine.agents.values())
                current_track_agent = agents[0]
                self.main_camera.set_follow_lane(self.config["use_chase_camera_follow_lane"])
                self.main_camera.track(current_track_agent)
                if bev_cam:
                    self.main_camera.stop_track()
                    self.main_camera.set_bird_view_pos_hpr(current_track_agent.position)
                for name, sensor in self.engine.sensors.items():
                    if hasattr(sensor, "track") and name != "main_camera":
                        sensor.track(current_track_agent.origin, [0., 0.8, 1.5], [0, 0.59681, 0])

    def _get_reset_return(self, reset_info):
        scene_manager_before_step_infos = reset_info
        scene_manager_after_step_infos = self.engine.after_step()

        obses = {}
        done_infos = {}
        cost_infos = {}
        reward_infos = {}
        engine_info = merge_dicts(
            scene_manager_after_step_infos, scene_manager_before_step_infos, allow_new_keys=True, without_copy=True
        )
        for v_id, v in self.agents.items():
            self.observations[v_id].reset(self, v)
            obses[v_id] = self.observations[v_id].observe(v)
            _, reward_infos[v_id] = self.reward_function(v_id)
            _, done_infos[v_id] = self.done_function(v_id)
            _, cost_infos[v_id] = self.cost_function(v_id)

        step_infos = concat_step_infos([engine_info, done_infos, reward_infos, cost_infos])

        if self.is_multi_agent:
            return obses, step_infos
        else:
            return self._wrap_as_single_agent(obses), self._wrap_info_as_single_agent(step_infos)

    def _wrap_info_as_single_agent(self, data):
        agent_info = data.pop(next(iter(self.agents.keys())))
        data.update(agent_info)
        return data

    def _get_step_return(self, actions, engine_info):
        obses = {}
        done_infos = {}
        cost_infos = {}
        reward_infos = {}
        rewards = {}
        for v_id, v in self.agents.items():
            self.episode_lengths[v_id] += 1
            rewards[v_id], reward_infos[v_id] = self.reward_function(v_id)
            self.episode_rewards[v_id] += rewards[v_id]
            done_function_result, done_infos[v_id] = self.done_function(v_id)
            _, cost_infos[v_id] = self.cost_function(v_id)
            self.dones[v_id] = done_function_result or self.dones[v_id]
            o = self.observations[v_id].observe(v)
            obses[v_id] = o

        step_infos = concat_step_infos([engine_info, done_infos, reward_infos, cost_infos])
        truncateds = {k: step_infos[k].get(TerminationState.MAX_STEP, False) for k in self.agents.keys()}
        terminateds = {k: self.dones[k] for k in self.agents.keys()}

        if self.config["horizon"] and self.episode_step > 5 * self.config["horizon"]:
            for k in truncateds:
                truncateds[k] = True
                if self.config["truncate_as_terminate"]:
                    self.dones[k] = terminateds[k] = True

        for v_id, r in rewards.items():
            step_infos[v_id]["episode_reward"] = self.episode_rewards[v_id]
            step_infos[v_id]["episode_length"] = self.episode_lengths[v_id]

        if not self.is_multi_agent:
            return self._wrap_as_single_agent(obses), self._wrap_as_single_agent(rewards), \
                   self._wrap_as_single_agent(terminateds), self._wrap_as_single_agent(
                truncateds), self._wrap_info_as_single_agent(step_infos)
        else:
            return obses, rewards, terminateds, truncateds, step_infos

    def close(self):
        if self.engine is not None:
            close_engine()

    def force_close(self):
        print("Closing environment ... Please wait")
        self.close()
        time.sleep(2)
        raise KeyboardInterrupt("'Esc' is pressed. MetaDrive exits now.")

    def capture(self, file_name=None):
        if not hasattr(self, "_capture_img"):
            self._capture_img = PNMImage()
        self.engine.win.getScreenshot(self._capture_img)
        if file_name is None:
            file_name = "main_index_{}_step_{}_{}.png".format(self.current_seed, self.engine.episode_step, time.time())
        self._capture_img.write(file_name)
        self.logger.info("Image is saved at: {}".format(file_name))

    def for_each_agent(self, func, *args, **kwargs):
        return self.agent_manager.for_each_active_agents(func, *args, **kwargs)

    def get_single_observation(self):
        if self.__class__ is BaseEnv:
            o = DummyObservation()
        else:
            if self.config["agent_observation"]:
                o = self.config["agent_observation"](self.config)
            else:
                img_obs = self.config["image_observation"]
                o = ImageStateObservation(self.config) if img_obs else LidarStateObservation(self.config)
        return o

    def _wrap_as_single_agent(self, data):
        return data[next(iter(self.agents.keys()))]

    def seed(self, seed=None):
        if seed is not None:
            set_global_random_seed(seed)

    @property
    def current_seed(self):
        return self.engine.global_random_seed

    @property
    def observations(self):
        return self.agent_manager.get_observations()

    @property
    def observation_space(self) -> gym.Space:
        ret = self.agent_manager.get_observation_spaces()
        if not self.is_multi_agent:
            return next(iter(ret.values()))
        else:
            return gym.spaces.Dict(ret)

    @property
    def action_space(self) -> gym.Space:
        ret = self.agent_manager.get_action_spaces()
        if not self.is_multi_agent:
            return next(iter(ret.values()))
        else:
            return gym.spaces.Dict(ret)

    @property
    def vehicles(self):
        self.logger.warning("env.vehicles will be deprecated soon. Use env.agents instead", extra={"log_once": True})
        return self.agents

    @property
    def vehicle(self):
        self.logger.warning("env.vehicle will be deprecated soon. Use env.agent instead", extra={"log_once": True})
        return self.agent

    @property
    def agents(self):
        return self.agent_manager.active_agents

    @property
    def agent(self):
        assert len(self.agents) == 1, (
            "env.agent is only supported in single-agent environment!"
            if len(self.agents) > 1 else "Please initialize the environment first!"
        )
        return self.agents[DEFAULT_AGENT]

    @property
    def agents_including_just_terminated(self):
        ret = self.agent_manager.active_agents
        ret.update(self.agent_manager.just_terminated_agents)
        return ret

    def setup_engine(self):
        self.engine.accept("r", self.reset)
        self.engine.accept("c", self.capture)
        self.engine.accept("p", self.stop)
        self.engine.accept("b", self.switch_to_top_down_view)
        self.engine.accept("q", self.switch_to_third_person_view)
        self.engine.accept("]", self.next_seed_reset)
        self.engine.accept("[", self.last_seed_reset)
        self.engine.register_manager("agent_manager", self.agent_manager)
        self.engine.register_manager("record_manager", RecordManager())
        self.engine.register_manager("replay_manager", ReplayManager())

    @property
    def current_map(self):
        return self.engine.current_map

    @property
    def maps(self):
        return self.engine.map_manager.maps

    def _render_topdown(self, text, *args, **kwargs):
        return self.engine.render_topdown(text, *args, **kwargs)

    @property
    def main_camera(self):
        return self.engine.main_camera

    @property
    def current_track_agent(self):
        return self.engine.current_track_agent

    @property
    def top_down_renderer(self):
        return self.engine.top_down_renderer

    @property
    def episode_step(self):
        return self.engine.episode_step if self.engine is not None else 0

    def export_scenarios(
        self,
        policies: Union[dict, Callable],
        scenario_index: Union[list, int],
        max_episode_length=None,
        verbose=False,
        suppress_warning=False,
        render_topdown=False,
        return_done_info=True,
        to_dict=True
    ):
        def _act(observation):
            if isinstance(policies, dict):
                ret = {}
                for id, o in observation.items():
                    ret[id] = policies[id](o)
            else:
                ret = policies(observation)
            return ret

        if self.is_multi_agent:
            assert isinstance(policies, dict), "In MARL setting, policies should be mapped to agents according to id"
        else:
            assert isinstance(policies, Callable), "In single agent case, policy should be a callable object, taking" \
                                                   "observation as input."
        scenarios_to_export = dict()
        if isinstance(scenario_index, int):
            scenario_index = [scenario_index]
        self.config["record_episode"] = True
        done_info = {}
        for index in scenario_index:
            obs = self.reset(seed=index)
            done = False
            count = 0
            info = None
            while not done:
                obs, reward, terminated, truncated, info = self.step(_act(obs))
                done = terminated or truncated
                count += 1
                if max_episode_length is not None and count > max_episode_length:
                    done = True
                    info[TerminationState.MAX_STEP] = True
                if count > 10000 and not suppress_warning:
                    self.logger.warning(
                        "Episode length is too long! If this behavior is intended, "
                        "set suppress_warning=True to disable this message"
                    )
                if render_topdown:
                    self.render("topdown")
            episode = self.engine.dump_episode()
            if verbose:
                self.logger.info("Finish scenario {} with {} steps.".format(index, count))
            scenarios_to_export[index] = convert_recorded_scenario_exported(episode, to_dict=to_dict)
            done_info[index] = info
        self.config["record_episode"] = False
        if return_done_info:
            return scenarios_to_export, done_info
        else:
            return scenarios_to_export

    def stop(self):
        self.in_stop = not self.in_stop

    def switch_to_top_down_view(self):
        self.main_camera.stop_track()

    def switch_to_third_person_view(self):
        if self.main_camera is None:
            return
        self.main_camera.reset()
        if self.config["prefer_track_agent"] is not None and self.config["prefer_track_agent"] in self.agents.keys():
            new_v = self.agents[self.config["prefer_track_agent"]]
            current_track_agent = new_v
        else:
            if self.main_camera.is_bird_view_camera():
                current_track_agent = self.current_track_agent
            else:
                agents = list(self.engine.agents.values())
                if len(agents) <= 1:
                    return
                if self.current_track_agent in agents:
                    agents.remove(self.current_track_agent)
                new_v = get_np_random().choice(agents)
                current_track_agent = new_v
        self.main_camera.track(current_track_agent)
        for name, sensor in self.engine.sensors.items():
            if hasattr(sensor, "track") and name != "main_camera":
                camera_video_posture = [0, 0.59681, 0]
                sensor.track(current_track_agent.origin, constants.DEFAULT_SENSOR_OFFSET, camera_video_posture)
        return

    def next_seed_reset(self):
        if self.current_seed + 1 < self.start_index + self.num_scenarios:
            self.reset(self.current_seed + 1)
        else:
            self.logger.warning(
                "Can't load next scenario! Current seed is already the max scenario index."
                "Allowed index: {}-{}".format(self.start_index, self.start_index + self.num_scenarios - 1)
            )

    def last_seed_reset(self):
        if self.current_seed - 1 >= self.start_index:
            self.reset(self.current_seed - 1)
        else:
            self.logger.warning(
                "Can't load last scenario! Current seed is already the min scenario index"
                "Allowed index: {}-{}".format(self.start_index, self.start_index + self.num_scenarios - 1)
            )

    def _reset_global_seed(self, force_seed=None):
        current_seed = force_seed if force_seed is not None else \
            get_np_random(self._DEBUG_RANDOM_SEED).randint(self.start_index, self.start_index + self.num_scenarios)
        assert self.start_index <= current_seed < self.start_index + self.num_scenarios, \
            "scenario_index (seed) should be in [{}:{})".format(self.start_index, self.start_index + self.num_scenarios)
        self.seed(current_seed)


if __name__ == '__main__':
    cfg = {"use_render": True}
    env = BaseEnv(cfg)
    env.reset()

    # Define initial action
    action = [0.0, 0.0]

    def on_press(key):
        global action
        try:
            if key.char == 'w':
                action[1] = 1.0  # Accelerate
            elif key.char == 's':
                action[1] = -1.0  # Brake
            elif key.char == 'a':
                action[0] = -1.0  # Steer left
            elif key.char == 'd':
                action[0] = 1.0  # Steer right
        except AttributeError:
            pass

    def on_release(key):
        global action
        if key.char in ['w', 's']:
            action[1] = 0.0  # Stop acceleration/brake
        elif key.char in ['a', 'd']:
            action[0] = 0.0  # Center steering

    # Initialize keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    while True:
        env.logger.info(f"Applying action: {action}")
        obs, reward, done, truncated, info = env.step(action)
        env.logger.info(f"Observation: {obs}, Reward: {reward}, Done: {done}, Truncated: {truncated}, Info: {info}")
        env.render()
        time.sleep(0.1)  # Adjust sleep time for better control responsiveness
        if done or truncated:
            env.logger.info("Episode finished")
            break
