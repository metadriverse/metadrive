import sys
import time
from collections import defaultdict
from typing import Union, Dict, AnyStr, Optional, Tuple

import gym
import numpy as np
from panda3d.core import PNMImage

from metadrive.component.blocks.first_block import FirstPGBlock
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import RENDER_MODE_NONE, DEFAULT_AGENT
from metadrive.engine.base_engine import BaseEngine
from metadrive.engine.engine_utils import initialize_engine, close_engine, \
    engine_initialized, set_global_random_seed
from metadrive.manager.agent_manager import AgentManager
from metadrive.obs.observation_base import ObservationBase
from metadrive.utils import Config, merge_dicts, get_np_random, concat_step_infos
from metadrive.utils.utils import auto_termination

BASE_DEFAULT_CONFIG = dict(
    # ===== Generalization =====
    start_seed=0,
    environment_num=1,

    # ===== agent =====
    random_agent_model=False,
    IDM_agent=False,

    # ===== multi-agent =====
    num_agents=1,  # Note that this can be set to >1 in MARL envs, or set to -1 for as many vehicles as possible.
    is_multi_agent=False,
    allow_respawn=False,
    delay_done=0,  # How many steps for the agent to stay static at the death place after done.

    # ===== Action =====
    manual_control=False,
    controller="keyboard",  # "joystick" or "keyboard"
    decision_repeat=5,
    discrete_action=False,
    discrete_steering_dim=5,
    discrete_throttle_dim=5,

    # ===== Rendering =====
    use_render=False,  # pop a window to render or not
    debug=False,
    disable_model_compression=True,  # disable compression if you wish to launch the window quicker.
    cull_scene=True,  # only for debug use
    use_chase_camera_follow_lane=False,  # If true, then vision would be more stable.
    camera_height=1.8,
    camera_dist=6.0,
    prefer_track_agent=None,
    draw_map_resolution=1024,  # Drawing the map in a canvas of (x, x) pixels.
    top_down_camera_initial_x=0,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=200,  # height

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
        use_special_color=False,

        # ===== use image =====
        image_source="rgb_camera",  # take effect when only when offscreen_render == True

        # ===== vehicle spawn and destination =====
        spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0),
        spawn_longitude=5.0,
        spawn_lateral=0.0,
        destination_node=None,

        # ==== others ====
        overtake_stat=False,  # we usually set to True when evaluation
        action_check=False,
        random_color=False,
    ),

    # ===== Engine Core config =====
    window_size=(1200, 900),  # width, height
    physics_world_step_size=2e-2,
    show_fps=True,
    global_light=False,
    # only render physics world without model, a special debug option
    debug_physics_world=False,
    # debug static world
    debug_static_world=False,
    # set to true only when on headless machine and use rgb image!!!!!!
    headless_machine_render=False,
    # turn on to profile the efficiency
    pstats=False,

    # ===== Others =====
    # The maximum distance used in PGLOD. Set to None will use the default values.
    max_distance=None,
    # Force to generate objects in the left lane.
    _debug_crash_object=False,
    record_episode=False,
    horizon=None,  # The maximum length of each episode. Set to None to remove this constraint
)


class BaseEnv(gym.Env):
    # Force to use this seed if necessary. Note that the recipient of the forced seed should be explicitly implemented.
    _DEBUG_RANDOM_SEED = None
    DEFAULT_AGENT = DEFAULT_AGENT

    @classmethod
    def default_config(cls) -> "Config":
        return Config(BASE_DEFAULT_CONFIG)

    # ===== Intialization =====
    def __init__(self, config: dict = None):
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
        self._top_down_renderer = None
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
                v_id: BaseVehicle.get_action_space_before_init(
                    self.config["vehicle_config"]["extra_action_dim"], self.config["discrete_action"],
                    self.config["discrete_steering_dim"], self.config["discrete_throttle_dim"]
                )
                for v_id in self.config["target_vehicle_configs"].keys()
            }
        else:
            return {
                DEFAULT_AGENT: BaseVehicle.get_action_space_before_init(
                    self.config["vehicle_config"]["extra_action_dim"], self.config["discrete_action"],
                    self.config["discrete_steering_dim"], self.config["discrete_throttle_dim"]
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
        actions = self._preprocess_actions(actions)
        step_infos = self._step_simulator(actions)
        o, r, d, i = self._get_step_return(actions, step_infos)
        return o, r, d, i

    def _preprocess_actions(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray]]) \
            -> Union[np.ndarray, Dict[AnyStr, np.ndarray]]:
        if not self.is_multi_agent:
            actions = {v_id: actions for v_id in self.vehicles.keys()}
        else:
            if self.config["vehicle_config"]["action_check"]:
                # Check whether some actions are not provided.
                given_keys = set(actions.keys())
                have_keys = set(self.vehicles.keys())
                assert given_keys == have_keys, "The input actions: {} have incompatible keys with existing {}!".format(
                    given_keys, have_keys
                )
            else:
                # That would be OK if extra actions is given. This is because, when evaluate a policy with naive
                # implementation, the "termination observation" will still be given in T=t-1. And at T=t, when you
                # collect action from policy(last_obs) without masking, then the action for "termination observation"
                # will still be computed. We just filter it out here.
                actions = {v_id: actions[v_id] for v_id in self.vehicles.keys()}
        return actions

    def _step_simulator(self, actions):
        # Note that we use shallow update for info dict in this function! This will accelerate system.
        scene_manager_before_step_infos = self.engine.before_step(actions)
        # step all entities
        self.engine.step(self.config["decision_repeat"])
        # update states, if restore from episode data, position and heading will be force set in update_state() function
        scene_manager_after_step_infos = self.engine.after_step()
        return merge_dicts(
            scene_manager_after_step_infos, scene_manager_before_step_infos, allow_new_keys=True, without_copy=True
        )

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

    def render(self, mode='human', text: Optional[Union[dict, str]] = None, *args, **kwargs) -> Optional[np.ndarray]:
        """
        This is a pseudo-render function, only used to update onscreen message when using panda3d backend
        :param mode: 'rgb'/'human'
        :param text:text to show
        :return: when mode is 'rgb', image array is returned
        """
        if mode == "top_down":
            return self._render_topdown(*args, **kwargs)
        assert self.config["use_render"] or self.engine.mode != RENDER_MODE_NONE, ("render is off now, can not render")
        self.engine.render_frame(text)
        if mode != "human" and self.config["offscreen_render"]:
            # fetch img from img stack to be make this func compatible with other render func in RL setting
            return self.vehicle.observations.img_obs.get_image()

        if mode == "rgb_array" and self.config["use_render"]:
            if not hasattr(self, "_temporary_img_obs"):
                from metadrive.obs.observation_base import ImageObservation
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
        self.engine.reset()
        if self._top_down_renderer is not None:
            self._top_down_renderer.reset(self.current_map)

        self.dones = {agent_id: False for agent_id in self.vehicles.keys()}
        self.episode_steps = 0
        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)
        assert (len(self.vehicles) == self.num_agents) or (self.num_agents == -1)

        return self._get_reset_return()

    def _get_reset_return(self):
        ret = {}
        self.engine.after_step()
        for v_id, v in self.vehicles.items():
            self.observations[v_id].reset(self, v)
            ret[v_id] = self.observations[v_id].observe(v)
        return ret if self.is_multi_agent else self._wrap_as_single_agent(ret)

    def _get_step_return(self, actions, step_infos):
        # update obs, dones, rewards, costs, calculate done at first !
        obses = {}
        done_infos = {}
        cost_infos = {}
        reward_infos = {}
        rewards = {}
        for v_id, v in self.vehicles.items():
            obses[v_id] = self.observations[v_id].observe(v)
            done_function_result, done_infos[v_id] = self.done_function(v_id)
            rewards[v_id], reward_infos[v_id] = self.reward_function(v_id)
            _, cost_infos[v_id] = self.cost_function(v_id)
            done = done_function_result or self.dones[v_id]
            self.dones[v_id] = done

        should_done = self.config["horizon"] and self.episode_steps >= self.config["horizon"]
        termination_infos = self.for_each_vehicle(auto_termination, should_done)

        step_infos = concat_step_infos([
            step_infos,
            done_infos,
            reward_infos,
            cost_infos,
            termination_infos,
        ])

        if should_done:
            for k in self.dones:
                self.dones[k] = True

        dones = {k: self.dones[k] for k in self.vehicles.keys()}
        for v_id, r in rewards.items():
            self.episode_rewards[v_id] += r
            step_infos[v_id]["episode_reward"] = self.episode_rewards[v_id]
            self.episode_lengths[v_id] += 1
            step_infos[v_id]["episode_length"] = self.episode_lengths[v_id]
        if not self.is_multi_agent:
            return self._wrap_as_single_agent(obses), self._wrap_as_single_agent(rewards), \
                   self._wrap_as_single_agent(dones), self._wrap_as_single_agent(step_infos)
        else:
            return obses, rewards, dones, step_infos

    def close(self):
        if self.engine is not None:
            close_engine()
        if self._top_down_renderer is not None:
            self._top_down_renderer.close()

    def force_close(self):
        print("Closing environment ... Please wait")
        self.close()
        time.sleep(2)  # Sleep two seconds
        raise KeyboardInterrupt("'Esc' is pressed. MetaDrive exits now.")

    def capture(self):
        img = PNMImage()
        self.engine.win.getScreenshot(img)
        img.write("main_{}.png".format(time.time()))

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
        self.engine.register_manager("agent_manager", self.agent_manager)

    @property
    def current_map(self):
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

    def _render_topdown(self, *args, **kwargs):
        if self._top_down_renderer is None:
            from metadrive.obs.top_down_renderer import TopDownRenderer
            self._top_down_renderer = TopDownRenderer(self, self.current_map, *args, **kwargs)
        return self._top_down_renderer.render(list(self.vehicles.values()), self.agent_manager)

    @property
    def main_camera(self):
        return self.engine.main_camera

    @property
    def current_track_vehicle(self):
        return self.engine.current_track_vehicle
