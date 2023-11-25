import logging
from metadrive.obs.image_obs import ImageStateObservation
from metadrive.obs.state_obs import LidarStateObservation
import time
from collections import defaultdict
from typing import Union, Dict, AnyStr, Optional, Tuple, Callable

import gymnasium as gym
import numpy as np
from panda3d.core import PNMImage

from metadrive.component.sensors.base_camera import BaseCamera
from metadrive.component.sensors.dashboard import DashBoard
from metadrive.component.sensors.distance_detector import LaneLineDetector, SideDetector
from metadrive.component.sensors.lidar import Lidar
from metadrive.constants import RENDER_MODE_NONE, DEFAULT_AGENT
from metadrive.constants import RENDER_MODE_ONSCREEN, RENDER_MODE_OFFSCREEN
from metadrive.constants import TerminationState
from metadrive.engine.engine_utils import initialize_engine, close_engine, \
    engine_initialized, set_global_random_seed, initialize_global_config, get_global_config
from metadrive.engine.logger import get_logger, set_log_level
from metadrive.manager.agent_manager import AgentManager
from metadrive.manager.record_manager import RecordManager
from metadrive.manager.replay_manager import ReplayManager
from metadrive.obs.observation_base import DummyObservation
from metadrive.obs.observation_base import ObservationBase
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.scenario.utils import convert_recorded_scenario_exported
from metadrive.utils import Config, merge_dicts, get_np_random, concat_step_infos
from metadrive.version import VERSION

BASE_DEFAULT_CONFIG = dict(

    # ===== agent =====
    random_agent_model=False,  # randomize the car model for the agent
    target_vehicle_configs={DEFAULT_AGENT: dict(use_special_color=False, spawn_lane_index=None)},

    # ===== multi-agent =====
    num_agents=1,  # Note that this can be set to >1 in MARL envs, or set to -1 for as many vehicles as possible.
    is_multi_agent=False,
    allow_respawn=False,
    delay_done=0,  # How many steps for the agent to stay static at the death place after done.

    # ===== Action/Control =====
    manual_control=False,  # if set to True, agent policy will be Manual Control policy
    controller="keyboard",  # "joystick" or "keyboard"
    agent_policy=EnvInputPolicy,
    discrete_action=False,
    discrete_steering_dim=5,
    discrete_throttle_dim=5,
    extra_action_dim=0,  # If you want to control more things besides throttle brake, set extra_action_dim
    use_multi_discrete=False,  # If True, use MultiDiscrete action space. Otherwise, use Discrete.
    action_check=False,

    # ===== Observation =====
    rgb_clip=True,  # clip rgb to (0, 1)
    stack_size=3,  # the number of timesteps for stacking image observation
    image_observation=False,  # use image observation or lidar
    agent_observation=None,

    # ===== Termination =====
    horizon=None,  # The maximum length of each environmental episode. Set to None to remove this constraint
    truncate_as_terminate=False,

    # ===== Main Camera =====
    use_chase_camera_follow_lane=False,  # If true, then vision would be more stable.
    camera_height=2.2,
    camera_dist=7.5,
    camera_pitch=None,  # degree
    camera_smooth=True,
    camera_smooth_buffer_size=20,
    camera_fov=65,
    prefer_track_agent=None,
    top_down_camera_initial_x=0,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=200,  # height

    # ===== Vehicle =====
    vehicle_config=dict(
        vehicle_model="default",
        show_navi_mark=True,
        enable_reverse=False,
        show_dest_mark=False,
        show_line_to_dest=False,
        show_line_to_navi_mark=False,
        use_special_color=False,
        no_wheel_friction=False,

        # ===== use image =====
        image_source="rgb_camera",  # take effect when only when image_observation == True

        # ===== vehicle spawn and destination =====
        navigation_module=None,  # a class type for self-defined navigation
        spawn_lane_index=None,
        spawn_longitude=5.0,
        spawn_lateral=0.0,
        destination=None,
        spawn_position_heading=None,
        spawn_velocity=None,  # m/s
        spawn_velocity_car_frame=False,

        # ==== others ====
        overtake_stat=False,  # we usually set to True when evaluation
        random_color=False,
        # The shape of vehicle are predefined by its class. But in special scenario (WaymoVehicle) we might want to
        # set to arbitrary shape.
        width=None,
        length=None,
        height=None,
        mass=None,

        # only for top-down drawing, the physics size wouldn't be changed
        top_down_width=None,
        top_down_length=None,

        # ===== vehicle module config =====
        lidar=dict(
            num_lasers=240, distance=50, num_others=0, gaussian_noise=0.0, dropout_prob=0.0, add_others_navi=False
        ),
        side_detector=dict(num_lasers=0, distance=50, gaussian_noise=0.0, dropout_prob=0.0),
        lane_line_detector=dict(num_lasers=0, distance=20, gaussian_noise=0.0, dropout_prob=0.0),
        show_lidar=False,
        show_side_detector=False,
        show_lane_line_detector=False,
        gaussian_noise=0.0,
        dropout_prob=0.0,
        light=False,  # vehicle light, only available when enabling render-pipeline
    ),

    # ===== Sensors =====
    # It should be a dict like this:
    # sensors={
    #   "rgb_camera": (RGBCamera, arg_1, arg_2, ..., arg_n),
    #    ...
    #   "your_sensor_name": (sensor_class, arg_1, arg_2, ..., arg_n)
    # }
    # Example:
    # sensors = dict(
    #           lidar=(Lidar,),
    #           side_detector=(SideDetector,),
    #           lane_line_detector=(LaneLineDetector,)
    #           rgb_camera=(RGBCamera, 84, 84),
    #           mini_map=(MiniMap, 84, 84, 250),
    #           depth_camera=(DepthCamera, 84, 84),
    #         )
    # These sensors will be constructed automatically and can be accessed in engine.get_sensor("sensor_name")
    # NOTE: main_camera will be added automatically if you are using offscreen/onscreen mode
    sensors=dict(lidar=(Lidar,), side_detector=(SideDetector,), lane_line_detector=(LaneLineDetector,)),

    # ===== Engine Core config =====
    # if true pop a window to render
    use_render=False,
    # or (width, height), if set to None, it will be automatically determined
    window_size=(1200, 900),
    # when main_camera is not the image_source for vehicle, reduce the window size to (1,1) for boosting efficiency
    auto_resize_window=True,
    # physics world step 0.02s * decision_repeat per env.step,
    physics_world_step_size=2e-2,
    decision_repeat=5,
    # this is an advanced feature for accessing image with moving them to ram!
    image_on_cuda=False,
    # We will determine the render mode automatically, it runs at physics-only mode by default
    _render_mode=RENDER_MODE_NONE,
    # None: unlimited, number: fps
    force_render_fps=None,
    # if set to True all objects will be force destroyed when call clear()
    force_destroy=False,
    # number of buffering objects for each class.
    # we will maintain a set of buffers in the engine to store the used objects and can reuse them
    # when possible. But it is possible that some classes of objects are always forcefully respawn
    # and thus those used objects are stored in the buffer and never be reused.
    num_buffering_objects=200,
    # turn on to use render pipeline, which provides advanced rendering effects (Beta)
    render_pipeline=False,
    # daytime is only available when using render-pipeline
    daytime="19:00",  # use string like "13:40", We usually set this by editor in toolkit
    # Shadow
    shadow_range=32,
    # Multi-thread rendering
    multi_thread_render=True,
    multi_thread_render_mode="Cull",  # or "Cull/Draw"
    # model optimization
    preload_models=True,  # preload pedestrian Object for avoiding lagging when creating it for the first time
    disable_model_compression=True,  # disable compression if you wish to launch the window quicker.

    # ===== Mesh Terrain =====
    # road will have a marin whose width is determined by this value, unit: [m]
    drivable_area_extension=7,
    # height scale for mountains, unit: [m]
    height_scale=50,
    # use plane collision or mesh collision
    use_mesh_terrain=False,
    # use full-size mesh terrain
    full_size_mesh=True,
    # show crosswalk
    show_crosswalk=False,
    # show sidewalk
    show_sidewalk=True,

    # ===== Debug =====
    pstats=False,  # turn on to profile the efficiency
    debug=False,  # debug, output more messages
    debug_panda3d=False,  # debug panda3d
    debug_physics_world=False,  # only render physics world without model, a special debug option
    debug_static_world=False,  # debug static world
    log_level=logging.INFO,  # log level. logging.DEBUG/logging.CRITICAL or so on
    show_coordinates=False,  # show coordinates for maps and objects for debug

    # ===== GUI =====
    show_fps=True,
    show_logo=True,
    show_mouse=True,
    show_skybox=True,
    show_terrain=True,
    show_interface=True,
    show_policy_mark=False,  # show marks for policies for debugging multi-policy setting
    show_interface_navi_mark=True,  # the arrow
    interface_panel=["dashboard"],  # a list whose element chosen from sensors.keys() + "dashboard"

    # ===== Record/Replay Metadata =====
    record_episode=False,  # when replay_episode is not None ,this option will be useless
    replay_episode=None,  # set the replay file to enable replay
    only_reset_when_replay=False,  # Scenario will only be initialized, while future trajectories will not be replayed
    force_reuse_object_name=False,  # If True, when restoring objects, use the same ID as in dataset
)


class BaseEnv(gym.Env):
    # Force to use this seed if necessary. Note that the recipient of the forced seed should be explicitly implemented.
    _DEBUG_RANDOM_SEED: Union[int, None] = None

    @classmethod
    def default_config(cls) -> Config:
        return Config(BASE_DEFAULT_CONFIG)

    # ===== Intialization =====
    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        self.logger = get_logger()
        set_log_level(config.get("log_level", logging.DEBUG if config.get("debug", False) else logging.INFO))
        merged_config = self.default_config().update(config, False, ["target_vehicle_configs", "sensors"])
        global_config = self._post_process_config(merged_config)

        self.config = global_config
        initialize_global_config(self.config)

        # agent check
        self.num_agents = self.config["num_agents"]
        self.is_multi_agent = self.config["is_multi_agent"]
        if not self.is_multi_agent:
            assert self.num_agents == 1
        else:
            assert not self.config["image_on_cuda"], "Image on cuda don't support Multi-agent!"
        assert isinstance(self.num_agents, int) and (self.num_agents > 0 or self.num_agents == -1)

        # observation and action space
        self.agent_manager = self._get_agent_manager()

        # lazy initialization, create the main vehicle in the lazy_init() func
        # self.engine: Optional[BaseEngine] = None

        # In MARL envs with respawn mechanism, varying episode lengths might happen.
        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)

        # press p to stop
        self.in_stop = False

        # scenarios
        self.start_index = 0
        self.num_scenarios = 1

    def _post_process_config(self, config):
        """Add more special process to merged config"""
        # Cancel interface panel
        self.logger.info("Environment: {}".format(self.__class__.__name__))
        self.logger.info("MetaDrive version: {}".format(VERSION))
        if not config["show_interface"]:
            config["interface_panel"] = []

        # Optimize main window
        if not config["use_render"] and config["image_observation"] and \
                config["vehicle_config"]["image_source"] != "main_camera" and config["auto_resize_window"]:
            # reduce size as we don't use the main camera content for improving efficiency
            config["window_size"] = (1, 1)
            self.logger.debug(
                "Main window size is reduced to (1, 1) for boosting efficiency."
                "To cancel this, set auto_resize_window = False"
            )

        # Optimize sensor creation in none-screen mode
        if not config["use_render"] and not config["image_observation"]:
            filtered = {}
            for id, cfg in config["sensors"].items():
                if not issubclass(cfg[0], BaseCamera):
                    filtered[id] = cfg
            config["sensors"] = filtered
            config["interface_panel"] = []

        # Merge dashboard config with sensors
        to_use = []
        if not config["render_pipeline"]:
            for panel in config["interface_panel"]:
                if panel == "dashboard":
                    config["sensors"]["dashboard"] = (DashBoard,)
                if panel not in config["sensors"]:
                    self.logger.warning(
                        "Fail to add sensor: {} to the interface. Remove it from panel list!".format(panel)
                    )
                else:
                    to_use.append(panel)
        config["interface_panel"] = to_use

        # Check sensor existence
        if config["use_render"] or config["image_observation"]:
            config["sensors"]["main_camera"] = ("MainCamera", *config["window_size"])

        # Merge default sensor to list
        sensor_cfg = self.default_config()["sensors"].update(config["sensors"])
        config["sensors"] = sensor_cfg

        # show sensor lists
        _str = "Sensors: [{}]"
        sensors_str = ""
        for _id, cfg in config["sensors"].items():
            sensors_str += "{}: {}{}, ".format(_id, cfg[0] if isinstance(cfg[0], str) else cfg[0].__name__, cfg[1:])
        self.logger.info(_str.format(sensors_str[:-2]))

        # determine render mode automatically
        if config["use_render"]:
            assert "main_camera" in config["sensors"]
            config["_render_mode"] = RENDER_MODE_ONSCREEN
        else:
            config["_render_mode"] = RENDER_MODE_NONE
            for sensor in config["sensors"].values():
                if sensor[0] == "MainCamera" or (issubclass(BaseCamera, sensor[0]) and sensor[0] != DashBoard):
                    config["_render_mode"] = RENDER_MODE_OFFSCREEN
                    break
        self.logger.info("Render Mode: {}".format(config["_render_mode"]))
        self.logger.info("Horizon (Max steps per agent): {}".format(config["horizon"]))
        if config["truncate_as_terminate"]:
            self.logger.warning(
                "When reaching max steps, both 'terminate' and 'truncate will be True."
                "Generally, only the `truncate` should be `True`"
            )
        return config

    def _get_observations(self) -> Dict[str, "ObservationBase"]:
        return {DEFAULT_AGENT: self.get_single_observation()}

    def _get_observation_space(self):
        return {v_id: obs.observation_space for v_id, obs in self.observations.items()}

    def _get_action_space(self):
        if self.is_multi_agent:
            return {
                v_id: self.config["agent_policy"].get_input_space()
                for v_id in self.config["target_vehicle_configs"].keys()
            }
        else:
            return {DEFAULT_AGENT: self.config["agent_policy"].get_input_space()}

    def _get_agent_manager(self):
        return AgentManager(init_observations=self._get_observations(), init_action_space=self._get_action_space())

    def lazy_init(self):
        """
        Only init once in runtime, variable here exists till the close_env is called
        :return: None
        """
        # It is the true init() func to create the main vehicle and its module, to avoid incompatible with ray
        if engine_initialized():
            return
        initialize_engine(self.config)
        # engine setup
        self.setup_engine()
        # other optional initialization
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

    # ===== Run-time =====
    def step(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray], int]):
        actions = self._preprocess_actions(actions)
        engine_info = self._step_simulator(actions)
        while self.in_stop:
            self.engine.taskMgr.step()
        return self._get_step_return(actions, engine_info=engine_info)

    def _preprocess_actions(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray], int]) \
            -> Union[np.ndarray, Dict[AnyStr, np.ndarray], int]:
        if not self.is_multi_agent:
            actions = {v_id: actions for v_id in self.vehicles.keys()}
        else:
            if self.config["action_check"]:
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
        self.logger.warning("Reward function is not implemented. Return reward = 0", extra={"log_once": True})
        return 0, {}

    def cost_function(self, vehicle_id: str) -> Tuple[float, Dict]:
        self.logger.warning("Cost function is not implemented. Return cost = 0", extra={"log_once": True})
        return 0, {}

    def done_function(self, vehicle_id: str) -> Tuple[bool, Dict]:
        self.logger.warning("Done function is not implemented. Return Done = False", extra={"log_once": True})
        return False, {}

    def render(self, text: Optional[Union[dict, str]] = None, mode=None, *args, **kwargs) -> Optional[np.ndarray]:
        """
        This is a pseudo-render function, only used to update onscreen message when using panda3d backend
        :param text: text to show
        :param mode: start_top_down rendering candidate parameter is ["top_down", "topdown", "bev", "birdview"]
        :return: None or top_down image
        """

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

        # if mode != "human" and self.config["image_observation"]:
        #     # fetch img from img stack to be make this func compatible with other render func in RL setting
        #     return self.observations[DEFAULT_AGENT].img_obs.get_image()
        #
        # if mode == "rgb_array":
        #     assert self.config["use_render"], "You should create a Panda3d window before rendering images!"
        #     # if not hasattr(self, "temporary_img_obs"):
        #     #     from metadrive.obs.image_obs import ImageObservation
        #     #     image_source = "rgb_camera"
        #     #     assert len(self.vehicles) == 1, "Multi-agent not supported yet!"
        #     #     self.temporary_img_obs = ImageObservation(self.vehicles[DEFAULT_AGENT].config, image_source, False)
        #     # # else:
        #     # #     raise ValueError("Not implemented yet!")
        #     # self.temporary_img_obs.observe(self.vehicles[DEFAULT_AGENT])
        #     # return self.temporary_img_obs.get_image()
        #     return self.engine._get_window_image(return_bytes=return_bytes)
        return None

    def reset(self, seed: Union[None, int] = None):
        """
        Reset the env, scene can be restored and replayed by giving episode_data
        Reset the environment or load an episode from episode data to recover is
        :param seed: The seed to set the env. It is actually the scenario index you intend to choose
        :return: None
        """
        if self.logger is None:
            self.logger = get_logger()
            log_level = self.config.get("log_level", logging.DEBUG if self.config.get("debug", False) else logging.INFO)
            set_log_level(log_level)
        self.lazy_init()  # it only works the first time when reset() is called to avoid the error when render
        self._reset_global_seed(seed)
        if self.engine is None:
            raise ValueError(
                "Current MetaDrive instance is broken. Please make sure there is only one active MetaDrive "
                "environment exists in one process. You can try to call env.close() and then call "
                "env.reset() to rescue this environment. However, a better and safer solution is to check the "
                "singleton of MetaDrive and restart your program."
            )
        self.engine.reset()
        if self.top_down_renderer is not None:
            self.top_down_renderer.reset(self.current_map)

        self.dones = {agent_id: False for agent_id in self.vehicles.keys()}
        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)

        assert (len(self.vehicles) == self.num_agents) or (self.num_agents == -1), \
            "Vehicles: {} != Num_agents: {}".format(len(self.vehicles), self.num_agents)
        assert self.config is self.engine.global_config is get_global_config(), "Inconsistent config may bring errors!"
        return self._get_reset_return()

    def _get_reset_return(self):
        # TODO: figure out how to get the information of the before step
        scene_manager_before_step_infos = {}
        scene_manager_after_step_infos = self.engine.after_step()

        obses = {}
        done_infos = {}
        cost_infos = {}
        reward_infos = {}
        engine_info = merge_dicts(
            scene_manager_after_step_infos, scene_manager_before_step_infos, allow_new_keys=True, without_copy=True
        )
        for v_id, v in self.vehicles.items():
            self.observations[v_id].reset(self, v)
            obses[v_id] = self.observations[v_id].observe(v)
            _, reward_infos[v_id] = self.reward_function(v_id)
            _, done_infos[v_id] = self.done_function(v_id)
            _, cost_infos[v_id] = self.cost_function(v_id)

        step_infos = concat_step_infos([engine_info, done_infos, reward_infos, cost_infos])

        if self.is_multi_agent:
            return (obses, step_infos)
        else:
            return (self._wrap_as_single_agent(obses), self._wrap_as_single_agent(step_infos))

    def _get_step_return(self, actions, engine_info):
        # update obs, dones, rewards, costs, calculate done at first !
        obses = {}
        done_infos = {}
        cost_infos = {}
        reward_infos = {}
        rewards = {}
        for v_id, v in self.vehicles.items():
            self.episode_lengths[v_id] += 1
            rewards[v_id], reward_infos[v_id] = self.reward_function(v_id)
            self.episode_rewards[v_id] += rewards[v_id]
            done_function_result, done_infos[v_id] = self.done_function(v_id)
            _, cost_infos[v_id] = self.cost_function(v_id)
            done = done_function_result or self.dones[v_id]
            self.dones[v_id] = done
            o = self.observations[v_id].observe(v)
            obses[v_id] = o

        step_infos = concat_step_infos([engine_info, done_infos, reward_infos, cost_infos])

        # For extreme scenario only. Force to terminate all vehicles if the environmental step exceeds 5 times horizon.
        should_external_done = False
        if self.config["horizon"] is not None:
            should_external_done = self.episode_step > 5 * self.config["horizon"]
        if should_external_done:
            for k in self.dones:
                self.dones[k] = True

        terminateds = {k: self.dones[k] for k in self.vehicles.keys()}
        truncateds = {k: step_infos[k][TerminationState.MAX_STEP] for k in self.vehicles.keys()}

        for v_id, r in rewards.items():
            step_infos[v_id]["episode_reward"] = self.episode_rewards[v_id]
            step_infos[v_id]["episode_length"] = self.episode_lengths[v_id]

        if not self.is_multi_agent:
            return self._wrap_as_single_agent(obses), self._wrap_as_single_agent(rewards), \
                   self._wrap_as_single_agent(terminateds), self._wrap_as_single_agent(
                truncateds), self._wrap_as_single_agent(step_infos)
        else:
            return obses, rewards, terminateds, truncateds, step_infos

    def close(self):
        if self.engine is not None:
            close_engine()
        # if self.logger is not None:
        #     self.logger.handlers.clear()
        #     del self.logger
        # self.logger = None

    def force_close(self):
        print("Closing environment ... Please wait")
        self.close()
        time.sleep(2)  # Sleep two seconds
        raise KeyboardInterrupt("'Esc' is pressed. MetaDrive exits now.")

    def capture(self, file_name=None):
        if not hasattr(self, "_capture_img"):
            self._capture_img = PNMImage()
        self.engine.win.getScreenshot(self._capture_img)
        if file_name is None:
            file_name = "main_index_{}_step_{}_{}.png".format(self.current_seed, self.engine.episode_step, time.time())
        self._capture_img.write(file_name)
        self.logger.info("Image is saved at: {}".format(file_name))

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

    def get_single_observation(self):
        if self.__class__ is BaseEnv:
            o = DummyObservation({})
        else:
            if self.config["agent_observation"]:
                o = self.config["agent_observation"](self.config)
            else:
                img_obs = self.config["image_observation"]
                o = ImageStateObservation(self.config) if img_obs else LidarStateObservation(self.config)
        return o

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

    # @property
    # def vehicles_including_just_terminated(self):
    #     """
    #     Return all vehicles that occupy some space in current environments
    #     :return: Dict[agent_id:vehicle]
    #     """
    #     ret = self.agent_manager.active_agents
    #     ret.update(self.agent_manager.just_terminated_agents)
    #     return ret

    def setup_engine(self):
        """
        Engine setting after launching
        """
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
    def current_track_vehicle(self):
        return self.engine.current_track_vehicle

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
        """
        We export scenarios into a unified format with 10hz sample rate
        """

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
        if self.config["prefer_track_agent"] is not None and self.config["prefer_track_agent"] in self.vehicles.keys():
            new_v = self.vehicles[self.config["prefer_track_agent"]]
            current_track_vehicle = new_v
        else:
            if self.main_camera.is_bird_view_camera():
                current_track_vehicle = self.current_track_vehicle
            else:
                vehicles = list(self.engine.agents.values())
                if len(vehicles) <= 1:
                    return
                if self.current_track_vehicle in vehicles:
                    vehicles.remove(self.current_track_vehicle)
                new_v = get_np_random().choice(vehicles)
                current_track_vehicle = new_v
        self.main_camera.track(current_track_vehicle)
        return

    def next_seed_reset(self):
        if self.current_seed + 1 < self.start_index + self.num_scenarios:
            self.reset(self.current_seed + 1)
        else:
            self.logger.warning("Can't load next scenario! current seed is already the max scenario index")

    def last_seed_reset(self):
        if self.current_seed - 1 >= self.start_index:
            self.reset(self.current_seed - 1)
        else:
            self.logger.warning("Can't load last scenario! current seed is already the min scenario index")

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
    while True:
        env.step(env.action_space.sample())
