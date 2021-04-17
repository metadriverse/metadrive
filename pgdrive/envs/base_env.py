import copy
import os.path as osp
import time
from collections import defaultdict
from typing import Union, Dict, AnyStr, Optional, Tuple

import gym
import numpy as np
from panda3d.core import PNMImage
from pgdrive.constants import RENDER_MODE_NONE, DEFAULT_AGENT
from pgdrive.obs.observation_type import ObservationType
from pgdrive.scene_creator.vehicle.base_vehicle import BaseVehicle
from pgdrive.scene_manager.scene_manager import SceneManager
from pgdrive.utils import PGConfig, merge_dicts
from pgdrive.world.pg_world import PGWorld

pregenerated_map_file = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "assets", "maps", "PGDrive-maps.json")

BASE_DEFAULT_CONFIG = dict(
    # ===== Generalization =====
    start_seed=0,
    environment_num=1,

    # ==== agents config =====
    num_agents=1,
    is_multi_agent=False,

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
    use_chase_camera=True,
    use_chase_camera_follow_lane=False,  # If true, then vision would be more stable.
    camera_height=1.8,

    # ===== Others =====
    pg_world_config=dict(
        window_size=(1200, 900),  # width, height
        physics_world_step_size=2e-2,
        show_fps=True,

        # show message when render is called
        onscreen_message=True,

        # limit the render fps
        # Press "f" to switch FPS, this config is deprecated!
        # force_fps=None,

        # only render physics world without model, a special debug option
        debug_physics_world=False,

        # set to true only when on headless machine and use rgb image!!!!!!
        headless_image=False,

        # turn on to profile the efficiency
        pstats=False,

        # The maximum distance used in PGLOD. Set to None will use the default values.
        max_distance=None,

        # Force to generate objects in the left lane.
        _debug_crash_object=False
    ),
    record_episode=False,
)


class BasePGDriveEnv(gym.Env):
    DEFAULT_AGENT = DEFAULT_AGENT

    @classmethod
    def default_config(cls) -> "PGConfig":
        return PGConfig(BASE_DEFAULT_CONFIG)

    # ===== Intialization =====
    def __init__(self, config: dict = None):
        self.default_config_copy = PGConfig(self.default_config(), unchangeable=True)
        merged_config = self._process_extra_config(config)
        self.config = self._post_process_config(merged_config)

        self.num_agents = self.config["num_agents"]
        self.is_multi_agent = self.config["is_multi_agent"]
        if not self.is_multi_agent:
            assert self.num_agents == 1
        assert isinstance(self.num_agents, int) and self.num_agents > 0

        # observation and action space
        self.observations = self._get_observations()
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

        # map setting
        self.start_seed = self.config["start_seed"]
        self.env_num = self.config["environment_num"]

        # lazy initialization, create the main vehicle in the lazy_init() func
        self.pg_world: Optional[PGWorld] = None
        self.scene_manager: Optional[SceneManager] = None
        self.main_camera = None
        self.controller = None
        self.restored_maps = dict()
        self.episode_steps = 0

        self.maps = {_seed: None for _seed in range(self.start_seed, self.start_seed + self.env_num)}
        self.current_seed = self.start_seed
        self.current_map = None

        self.vehicles = dict()
        self.done_vehicles = dict()
        self.dones = None
        self.episode_rewards = defaultdict(float)
        # In MARL envs with reborn mechanism, varying episode lengths might happen.
        self.episode_lengths = defaultdict(int)
        self._pending_force_seed = None

    def _process_extra_config(self, config: Union[dict, "PGConfig"]) -> "PGConfig":
        """Check, update, sync and overwrite some config."""
        return config

    def _post_process_config(self, config):
        """Add more special process to merged config"""
        return config

    def _get_observations(self) -> Dict[str, "ObservationType"]:
        raise NotImplementedError()

    def _get_observation_space(self) -> gym.Space:
        ret = gym.spaces.Dict({v_id: obs.observation_space for v_id, obs in self.observations.items()})
        if not self.is_multi_agent:
            ret = next(iter((ret.spaces.values())))
        return ret

    def _get_action_space(self) -> gym.Space:
        ret = gym.spaces.Dict({v_id: BaseVehicle.get_action_space_before_init() for v_id in self.observations.keys()})
        if not self.is_multi_agent:
            ret = next(iter((ret.spaces.values())))
        return ret

    def _setup_pg_world(self) -> "PGWorld":
        pg_world = PGWorld(self.config["pg_world_config"])
        return pg_world

    def _get_scene_manager(self) -> "SceneManager":
        traffic_config = {"traffic_mode": self.config["traffic_mode"], "random_traffic": self.config["random_traffic"]}
        manager = SceneManager(
            self.config, self.pg_world, traffic_config, self.config["record_episode"], self.config["cull_scene"]
        )
        return manager

    def lazy_init(self):
        """
        Only init once in runtime, variable here exists till the close_env is called
        :return: None
        """
        # It is the true init() func to create the main vehicle and its module
        if self.pg_world is not None:
            return

        # init world
        self.pg_world = self._setup_pg_world()

        # init traffic manager
        self.scene_manager = self._get_scene_manager()

        # init vehicle
        self.vehicles = self._get_vehicles()

        # other optional initialization
        self._after_lazy_init()

    def _get_vehicles(self):
        return {self.DEFAULT_AGENT: BaseVehicle(self.pg_world, self.config["vehicle_config"])}

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
        scene_manager_infos = self.scene_manager.prepare_step(actions)
        action_infos = merge_dicts(action_infos, scene_manager_infos, allow_new_keys=True, without_copy=True)

        # step all entities
        self.scene_manager.step(self.config["decision_repeat"])

        # update states, if restore from episode data, position and heading will be force set in update_state() function
        scene_manager_step_infos = self.scene_manager.update_state()
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
        assert self.config["use_render"] or self.pg_world.mode != RENDER_MODE_NONE, (
            "render is off now, can not render"
        )
        self.pg_world.render_frame(text)
        if mode != "human" and self.config["use_image"]:
            # fetch img from img stack to be make this func compatible with other render func in RL setting
            return self.vehicle.observations.img_obs.get_image()

        if mode == "rgb_array" and self.config["use_render"]:
            if not hasattr(self, "_temporary_img_obs"):
                from pgdrive.obs.observation_type import ImageObservation
                image_source = "rgb_cam"
                assert len(self.vehicles) == 1, "Multi-agent not supported yet!"
                self.temporary_img_obs = ImageObservation(
                    self.vehicles[DEFAULT_AGENT].vehicle_config, image_source, False
                )
            else:
                raise ValueError("Not implemented yet!")
            self.temporary_img_obs.observe(self.vehicles[DEFAULT_AGENT].image_sensors[image_source])
            return self.temporary_img_obs.get_image()

        # logging.warning("You do not set 'use_image' or 'use_image' to True, so no image will be returned!")
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
        self.pg_world.clear_world()
        self._update_map(episode_data, force_seed)

        self._reset_vehicles()

        self.dones = {agent_id: False for agent_id in self.vehicles.keys()}
        self.episode_steps = 0
        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)

        # generate new traffic according to the map
        self.scene_manager.reset(
            self.current_map,
            self.vehicles,
            self.config["traffic_density"],
            self.config["accident_prob"],
            episode_data=episode_data
        )

        if self.main_camera is not None:
            self.main_camera.reset()

        return self._get_reset_return()

    def _update_map(self, episode_data: Union[None, dict] = None, force_seed: Union[None, int] = None):
        raise NotImplementedError()

    def _reset_vehicles(self):
        raise NotImplementedError()

    def _get_reset_return(self):
        raise NotImplementedError()

    def close(self):
        if self.pg_world is not None:
            if self.main_camera is not None:
                self.main_camera.destroy(self.pg_world)
                del self.main_camera
                self.main_camera = None
            self.pg_world.clear_world()

            self.scene_manager.destroy(self.pg_world)
            del self.scene_manager
            self.scene_manager = None

            if self.vehicles:
                self.for_each_vehicle(lambda v: v.destroy(self.pg_world))
            del self.vehicles
            self.vehicles = dict()

            del self.controller
            self.controller = None

            self.pg_world.close_world()
            del self.pg_world
            self.pg_world = None

        del self.maps
        self.maps = {_seed: None for _seed in range(self.start_seed, self.start_seed + self.env_num)}
        del self.current_map
        self.current_map = None
        del self.restored_maps
        self.restored_maps = dict()

    def force_close(self):
        print("Closing environment ... Please wait")
        self.close()
        time.sleep(2)  # Sleep two seconds
        raise KeyboardInterrupt("'Esc' is pressed. PGDrive exits now.")

    def set_current_seed(self, seed):
        self.current_seed = seed

    def capture(self):
        img = PNMImage()
        self.pg_world.win.getScreenshot(img)
        img.write("main.jpg")

        for name, sensor in self.vehicle.image_sensors.items():
            if name == "mini_map":
                name = "lidar"
            sensor.save_image("{}.jpg".format(name))

    def for_each_vehicle(self, func, *args, **kwargs):
        """
        func is a function that take each vehicle as the first argument and *arg and **kwargs as others.
        """
        assert len(self.vehicles) > 0, "No vehicles exist!"
        ret = dict()
        for k, v in self.vehicles.items():
            ret[k] = func(v, *args, **kwargs)
        return ret

    @property
    def vehicle(self):
        """A helper to return the vehicle only in the single-agent environment!"""
        assert len(self.vehicles) == 1, "env.vehicle is only supported in single-agent environment!"
        ego_v = self.vehicles[DEFAULT_AGENT]
        return ego_v

    def get_single_observation(self, vehicle_config: "PGConfig") -> "ObservationType":
        raise NotImplementedError()

    def _wrap_as_single_agent(self, data):
        return data[next(iter(self.vehicles.keys()))]

    def seed(self, seed=None):
        if seed:
            self._pending_force_seed = seed
