import os.path as osp
import sys
import time
from collections import defaultdict
from typing import Union, Dict, AnyStr, Optional, Tuple

import gym
import numpy as np
from panda3d.core import PNMImage
from pgdrive.component.vehicle.base_vehicle import BaseVehicle
from pgdrive.constants import RENDER_MODE_NONE, DEFAULT_AGENT
from pgdrive.engine.base_engine import BaseEngine
from pgdrive.engine.engine_utils import initialize_engine, close_engine, \
    engine_initialized, set_global_random_seed
from pgdrive.manager.agent_manager import AgentManager
from pgdrive.obs.observation_base import ObservationBase
from pgdrive.utils import Config, merge_dicts, get_np_random

BASE_DEFAULT_CONFIG = dict(
    # ===== Generalization =====
    start_seed=0,
    environment_num=1,

    # ==== agents config =====
    num_agents=1,  # Note that this can be set to >1 in MARL envs, or set to -1 for as many vehicles as possible.
    is_multi_agent=False,
    allow_respawn=False,
    delay_done=0,  # How many steps for the agent to stay static at the death place after done.
    random_agent_model=False,

    # ===== Action =====
    decision_repeat=5,

    # ===== Rendering =====
    use_render=False,  # pop a window to render or not
    # force_fps=None,
    debug=False,
    fast=False,  # disable compression if you wish to launch the window quicker.
    cull_scene=True,  # only for debug use
    manual_control=False,
    controller="keyboard",  # "joystick" or "keyboard"
    use_chase_camera_follow_lane=False,  # If true, then vision would be more stable.
    camera_height=1.8,
    camera_dist=7,
    prefer_track_agent=None,
    draw_map_resolution=1024,  # Drawing the map in a canvas of (x, x) pixels.
    top_down_camera_initial_x=0,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=120,  # height

    # ===== Vehicle =====
    vehicle_config=dict(
        increment_steering=False,
        vehicle_model="default",
        show_navi_mark=True,
        extra_action_dim=0,
        enable_reverse=False,
        random_navi_mark_color=False,
        show_dest_mark=False,
        show_line_to_dest=False,
        am_i_the_special_one=False
    ),

    # ===== Others =====
    window_size=(1200, 900),  # width, height
    physics_world_step_size=2e-2,
    show_fps=True,
    global_light=False,

    # show message when render is called
    onscreen_message=True,

    # limit the render fps
    # Press "f" to switch FPS, this config is deprecated!
    # force_fps=None,

    # only render physics world without model, a special debug option
    debug_physics_world=False,

    # debug static world
    debug_static_world=False,

    # set to true only when on headless machine and use rgb image!!!!!!
    headless_machine_render=False,

    # turn on to profile the efficiency
    pstats=False,

    # The maximum distance used in PGLOD. Set to None will use the default values.
    max_distance=None,

    # Force to generate objects in the left lane.
    _debug_crash_object=False,
    record_episode=False,
)


class BasePGDriveEnv(gym.Env):
    DEFAULT_AGENT = DEFAULT_AGENT

    # Force to use this seed if necessary. Note that the recipient of the forced seed should be explicitly implemented.
    _DEBUG_RANDOM_SEED = None

    @classmethod
    def default_config(cls) -> "Config":
        return Config(BASE_DEFAULT_CONFIG)

    # ===== Intialization =====
    def __init__(self, config: dict = None):
        self.default_config_copy = Config(self.default_config(), unchangeable=True)
        merged_config = self._merge_extra_config(config)
        global_config = self._post_process_config(merged_config)
        self.config = global_config

        # agent check
        self.num_agents = self.config["num_agents"]
        self.is_multi_agent = self.config["is_multi_agent"]
        if not self.is_multi_agent:
            assert self.num_agents == 1
        assert isinstance(self.num_agents, int) and (self.num_agents > 0 or self.num_agents == -1)

        # observation and action space
        self.agent_manager = AgentManager(
            init_observations=self._get_observations(), init_action_space=self._get_action_space()
        )

        # map setting
        self.start_seed = self.config["start_seed"]
        self.env_num = self.config["environment_num"]

        # lazy initialization, create the main vehicle in the lazy_init() func
        self.engine: Optional[BaseEngine] = None
        self.episode_steps = 0
        # self.current_seed = None

        # In MARL envs with respawn mechanism, varying episode lengths might happen.
        self.dones = None
        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)

    def _merge_extra_config(self, config: Union[dict, "Config"]) -> "Config":
        """Check, update, sync and overwrite some config."""
        return config

    def _post_process_config(self, config):
        """Add more special process to merged config"""
        return config

    def _get_observations(self) -> Dict[str, "ObservationBase"]:
        raise NotImplementedError()

    def _get_observation_space(self):
        return {v_id: obs.observation_space for v_id, obs in self.observations.items()}

    def _get_action_space(self):
        if self.is_multi_agent:
            return {
                v_id: BaseVehicle.get_action_space_before_init(self.config["vehicle_config"]["extra_action_dim"])
                for v_id in self.config["target_vehicle_configs"].keys()
            }
        else:
            return {
                DEFAULT_AGENT: BaseVehicle.get_action_space_before_init(
                    self.config["vehicle_config"]["extra_action_dim"]
                )
            }

    def lazy_init(self):
        """
        Only init once in runtime, variable here exists till the close_env is called
        :return: None
        """
        # It is the true init() func to create the main vehicle and its module, to avoid incompatible with ray
        if engine_initialized():
            return
        self.engine = initialize_engine(self.config)
        # engine setup
        self.setup_engine()
        # other optional initialization
        self._after_lazy_init()

    def _after_lazy_init(self):
        pass

    # ===== Run-time =====
    def step(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray]]):
        self.episode_steps += 1
        actions, action_infos = self._preprocess_actions(actions)
        step_infos = self._step_simulator(actions, action_infos)
        o, r, d, i = self._get_step_return(actions, step_infos)
        # return o, copy.deepcopy(r), copy.deepcopy(d), copy.deepcopy(i)
        return o, r, d, i

    def _preprocess_actions(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray]]) \
            -> Tuple[Union[np.ndarray, Dict[AnyStr, np.ndarray]], Dict]:
        raise NotImplementedError()

    def _step_simulator(self, actions, action_infos):
        # Note that we use shallow update for info dict in this function! This will accelerate system.
        scene_manager_infos = self.engine.before_step(actions)
        action_infos = merge_dicts(action_infos, scene_manager_infos, allow_new_keys=True, without_copy=True)

        # step all entities
        self.engine.step(self.config["decision_repeat"])

        # update states, if restore from episode data, position and heading will be force set in update_state() function
        scene_manager_step_infos = self.engine.after_step()
        action_infos = merge_dicts(action_infos, scene_manager_step_infos, allow_new_keys=True, without_copy=True)
        return action_infos

    def _get_step_return(self, actions, step_infos):
        """Return a tuple of obs, reward, dones, infos"""
        raise NotImplementedError()

    def reward_function(self, vehicle_id: str) -> Tuple[float, Dict]:
        """
        Override this func to get a new reward function
        :param vehicle_id: name of this base vehicle
        :return: reward, reward info
        """
        raise NotImplementedError()

    def cost_function(self, vehicle_id: str) -> Tuple[float, Dict]:
        raise NotImplementedError()

    def done_function(self, vehicle_id: str) -> Tuple[bool, Dict]:
        raise NotImplementedError()

    def render(self, mode='human', text: Optional[Union[dict, str]] = None) -> Optional[np.ndarray]:
        """
        This is a pseudo-render function, only used to update onscreen message when using panda3d backend
        :param mode: 'rgb'/'human'
        :param text:text to show
        :return: when mode is 'rgb', image array is returned
        """
        assert self.config["use_render"] or self.engine.mode != RENDER_MODE_NONE, ("render is off now, can not render")
        self.engine.render_frame(text)
        if mode != "human" and self.config["offscreen_render"]:
            # fetch img from img stack to be make this func compatible with other render func in RL setting
            return self.vehicle.observations.img_obs.get_image()

        if mode == "rgb_array" and self.config["use_render"]:
            if not hasattr(self, "_temporary_img_obs"):
                from pgdrive.obs.observation_base import ImageObservation
                image_source = "rgb_camera"
                assert len(self.vehicles) == 1, "Multi-agent not supported yet!"
                self.temporary_img_obs = ImageObservation(self.vehicles[DEFAULT_AGENT].config, image_source, False)
            else:
                raise ValueError("Not implemented yet!")
            self.temporary_img_obs.observe(self.vehicles[DEFAULT_AGENT].image_sensors[image_source])
            return self.temporary_img_obs.get_image()

        # logging.warning("You do not set 'offscreen_render' or 'offscreen_render' to True, so no image will be returned!")
        return None

    def reset(self, episode_data: dict = None, force_seed: Union[None, int] = None):
        """
        Reset the env, scene can be restored and replayed by giving episode_data
        Reset the environment or load an episode from episode data to recover is
        :param episode_data: Feed the episode data to replay an episode
        :param force_seed: The seed to set the env.
        :return: None
        """
        self.lazy_init()  # it only works the first time when reset() is called to avoid the error when render
        self._reset_global_seed(force_seed)
        self._update_map(episode_data=episode_data)

        self._reset_config()
        self.engine.reset()

        self.dones = {agent_id: False for agent_id in self.vehicles.keys()}
        self.episode_steps = 0
        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)

        return self._get_reset_return()

    def _update_map(self, episode_data: dict = None):
        self.engine.map_manager.update_map(self.config, self.current_seed, episode_data)

    # def _update_map(self, episode_data: Union[None, dict] = None):
    #     raise NotImplementedError()

    def _get_reset_return(self):
        raise NotImplementedError()

    def close(self):
        if self.engine is not None:
            close_engine()

    def force_close(self):
        print("Closing environment ... Please wait")
        self.close()
        time.sleep(2)  # Sleep two seconds
        raise KeyboardInterrupt("'Esc' is pressed. PGDrive exits now.")

    def capture(self):
        img = PNMImage()
        self.engine.win.getScreenshot(img)
        img.write("main.jpg")

        for name, sensor in self.vehicle.image_sensors.items():
            if name == "mini_map":
                name = "lidar"
            sensor.save_image("{}.jpg".format(name))

    def for_each_vehicle(self, func, *args, **kwargs):
        return self.agent_manager.for_each_active_agents(func, *args, **kwargs)

    @property
    def vehicle(self):
        """A helper to return the vehicle only in the single-agent environment!"""
        assert len(self.vehicles) == 1, (
            "env.vehicle is only supported in single-agent environment!"
            if len(self.vehicles) > 1 else "Please initialize the environment first!"
        )
        ego_v = self.vehicles[DEFAULT_AGENT]
        return ego_v

    def get_single_observation(self, vehicle_config: "Config") -> "ObservationBase":
        raise NotImplementedError()

    def _wrap_as_single_agent(self, data):
        return data[next(iter(self.vehicles.keys()))]

    def seed(self, seed=None):
        if seed is not None:
            set_global_random_seed(seed)

    @property
    def current_seed(self):
        return self.engine.global_random_seed

    @property
    def observations(self):
        """
        Return observations of active and controllable vehicles
        :return: Dict
        """
        return self.agent_manager.get_observations()

    @property
    def observation_space(self) -> gym.Space:
        """
        Return observation spaces of active and controllable vehicles
        :return: Dict
        """
        ret = self.agent_manager.get_observation_spaces()
        if not self.is_multi_agent:
            return next(iter(ret.values()))
        else:
            return gym.spaces.Dict(ret)

    @property
    def action_space(self) -> gym.Space:
        """
        Return observation spaces of active and controllable vehicles
        :return: Dict
        """
        ret = self.agent_manager.get_action_spaces()
        if not self.is_multi_agent:
            return next(iter(ret.values()))
        else:
            return gym.spaces.Dict(ret)

    @property
    def vehicles(self):
        """
        Return all active vehicles
        :return: Dict[agent_id:vehicle]
        """
        return self.agent_manager.active_agents

    def setup_engine(self):
        """
        Engine setting after launching
        """
        self.engine.accept("r", self.reset)
        self.engine.accept("escape", sys.exit)
        self.engine.accept("p", self.capture)
        from pgdrive.manager.map_manager import MapManager

        self.engine.register_manager("agent_manager", self.agent_manager)
        self.engine.register_manager("map_manager", MapManager())

    @property
    def current_map(self):
        # TODO(pzh): Can we remove this?
        return self.engine.map_manager.current_map

    def _reset_global_seed(self, force_seed):
        # create map
        if force_seed is not None:
            current_seed = force_seed
        else:
            current_seed = get_np_random(self._DEBUG_RANDOM_SEED
                                         ).randint(self.start_seed, self.start_seed + self.env_num)
        self.seed(current_seed)

    @property
    def maps(self):
        return self.engine.map_manager.pg_maps

    def _reset_config(self):
        """
        You may need to modify the global config in the new episode, do it here
        """
        pass
