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
from metadrive.constants import DEFAULT_SENSOR_HPR, DEFAULT_SENSOR_OFFSET
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

BASE_DEFAULT_CONFIG = dict(

    # ===== agent =====
    # Whether randomize the car model for the agent, randomly choosing from 4 types of cars
    random_agent_model=False,
    # The ego config is: env_config["vehicle_config"].update(env_config"[agent_configs"]["default_agent"])
    agent_configs={DEFAULT_AGENT: dict(use_special_color=True, spawn_lane_index=None)},

    # ===== multi-agent =====
    # This should be >1 in MARL envs, or set to -1 for spawning as many vehicles as possible.
    num_agents=1,
    # Turn on this to notify the simulator that it is MARL env
    is_multi_agent=False,
    # The number of agent will be fixed adn determined at the start of the episode, if set to False
    allow_respawn=False,
    # How many substeps for the agent to stay static at the death place after done. (Default for MARL: 25)
    delay_done=0,

    # ===== Action/Control =====
    # Please see Documentation: Action and Policy for more details
    # What policy to use for controlling agents
    agent_policy=EnvInputPolicy,
    # If set to True, agent_policy will be overriden and change to ManualControlPolicy
    manual_control=False,
    # What interfaces to use for manual control, options: "steering_wheel" or "keyboard" or "xbos"
    controller="keyboard",
    # Used with EnvInputPolicy. If set to True, the env.action_space will be discrete
    discrete_action=False,
    # If True, use MultiDiscrete action space. Otherwise, use Discrete.
    use_multi_discrete=False,
    # How many discrete actions are used for steering dim
    discrete_steering_dim=5,
    # How many discrete actions are used for throttle/brake dim
    discrete_throttle_dim=5,
    # Check if the action is contained in gym.space. Usually turned off to speed up simulation
    action_check=False,

    # ===== Observation =====
    # Please see Documentation: Observation for more details
    # Whether to normalize the pixel value from 0-255 to 0-1
    norm_pixel=True,
    # The number of timesteps for stacking image observation
    stack_size=3,
    # Whether to use image observation or lidar. It takes effect in get_single_observation
    image_observation=False,
    # Like agent_policy, users can use customized observation class through this field
    agent_observation=None,

    # ===== Termination =====
    # The maximum length of each agent episode. Set to None to remove this constraint
    horizon=None,
    # If set to True, the terminated will be True as well when the length of agent episode exceeds horizon
    truncate_as_terminate=False,

    # ===== Main Camera =====
    # A True value makes the camera follow the reference line instead of the vehicle, making its movement smooth
    use_chase_camera_follow_lane=False,
    # Height of the main camera
    camera_height=2.2,
    # Distance between the camera and the vehicle. It is the distance projecting to the x-y plane.
    camera_dist=7.5,
    # Pitch of main camera. If None, this will be automatically calculated
    camera_pitch=None,  # degree
    # Smooth the camera movement
    camera_smooth=True,
    # How many frames used to smooth the camera
    camera_smooth_buffer_size=20,
    # FOV of main camera
    camera_fov=65,
    # Only available in MARL setting, choosing which agent to track. Values should be "agent0", "agent1" or so on
    prefer_track_agent=None,
    # Setting the camera position for the Top-down Camera for 3D viewer (pressing key "B" to activate it)
    top_down_camera_initial_x=0,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=200,

    # ===== Vehicle =====
    vehicle_config=dict(
        # Vehicle model. Candidates: "s", "m", "l", "xl", "default". random_agent_model makes this config invalid
        vehicle_model="default",
        # If set to True, the vehicle can go backwards with throttle/brake < -1
        enable_reverse=False,
        # Whether to show the box as navigation points
        show_navi_mark=True,
        # Whether to show a box mark at the destination
        show_dest_mark=False,
        # Whether to draw a line from current vehicle position to the designation point
        show_line_to_dest=False,
        # Whether to draw a line from current vehicle position to the next navigation point
        show_line_to_navi_mark=False,
        # Whether to draw left / right arrow in the interface to denote the navigation direction
        show_navigation_arrow=True,
        # If set to True, the vehicle will be in color green in top-down renderer or MARL setting
        use_special_color=False,
        # Clear wheel friction, so it can not move by setting steering and throttle/brake. Used for ReplayPolicy
        no_wheel_friction=False,

        # ===== image capturing =====
        # Which camera to use for image observation. It should be a sensor registered in sensor config.
        image_source="rgb_camera",

        # ===== vehicle spawn and navigation =====
        # A BaseNavigation instance. It should match the road network type.
        navigation_module=None,
        # A lane id specifies which lane to spawn this vehicle
        spawn_lane_index=None,
        # destination lane id. Required only when navigation module is not None.
        destination=None,
        # the longitudinal and lateral position on the spawn lane
        spawn_longitude=5.0,
        spawn_lateral=0.0,

        # If the following items is assigned, the vehicle will be spawn at the specified position with certain speed
        spawn_position_heading=None,
        spawn_velocity=None,  # m/s
        spawn_velocity_car_frame=False,

        # ==== others ====
        # How many cars the vehicle has overtaken. It is deprecated due to bug.
        overtake_stat=False,
        # If set to True, the default texture for the vehicle will be replaced with a pure color one.
        random_color=False,
        # The shape of vehicle are predefined by its class. But in special scenario (WaymoVehicle) we might want to
        # set to arbitrary shape.
        width=None,
        length=None,
        height=None,
        mass=None,
        scale=None,  # triplet (x, y, z)

        # Set the vehicle size only for pygame top-down renderer. It doesn't affect the physical size!
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
        # Whether to turn on vehicle light, only available when enabling render-pipeline
        light=False,
    ),

    # ===== Sensors =====
    sensors=dict(lidar=(Lidar, ), side_detector=(SideDetector, ), lane_line_detector=(LaneLineDetector, )),

    # ===== Engine Core config =====
    # If true pop a window to render
    use_render=False,
    # (width, height), if set to None, it will be automatically determined
    window_size=(1200, 900),
    # Physics world step is 0.02s and will be repeated for decision_repeat times per env.step()
    physics_world_step_size=2e-2,
    decision_repeat=5,
    # This is an advanced feature for accessing image without moving them to ram!
    image_on_cuda=False,
    # Don't set this config. We will determine the render mode automatically, it runs at physics-only mode by default.
    _render_mode=RENDER_MODE_NONE,
    # If set to None: the program will run as fast as possible. Otherwise, the fps will be limited under this value
    force_render_fps=None,
    # We will maintain a set of buffers in the engine to store the used objects and can reuse them when possible
    # enhancing the efficiency. If set to True, all objects will be force destroyed when call clear()
    force_destroy=False,
    # Number of buffering objects for each class.
    num_buffering_objects=200,
    # Turn on it to use render pipeline, which provides advanced rendering effects (Beta)
    render_pipeline=False,
    # daytime is only available when using render-pipeline
    daytime="19:00",  # use string like "13:40", We usually set this by editor in toolkit
    # Shadow range, unit: [m]
    shadow_range=50,
    # Whether to use multi-thread rendering
    multi_thread_render=True,
    multi_thread_render_mode="Cull",  # or "Cull/Draw"
    # Model loading optimization. Preload pedestrian for avoiding lagging when creating it for the first time
    preload_models=True,
    # model compression increasing the launch time
    disable_model_compression=True,
    # Whether to disable the collision detection (useful for debugging / replay logged scenarios)
    disable_collision=False,

    # ===== Terrain =====
    # The size of the square map region, which is centered at [0, 0]. The map objects outside it are culled.
    map_region_size=2048,
    # Whether to remove lanes outside the map region. If True, lane localization only applies to map region
    cull_lanes_outside_map=False,
    # Road will have a flat marin whose width is determined by this value, unit: [m]
    drivable_area_extension=7,
    # Height scale for mountains, unit: [m]. 0 height makes the terrain flat
    height_scale=50,
    # If using mesh collision, mountains will have physics body and thus interact with vehicles.
    use_mesh_terrain=False,
    # If set to False, only the center region of the terrain has the physics body
    full_size_mesh=True,
    # Whether to show crosswalk
    show_crosswalk=True,
    # Whether to show sidewalk
    show_sidewalk=True,

    # ===== Debug =====
    # Please see Documentation: Debug for more details
    pstats=False,  # turn on to profile the efficiency
    debug=False,  # debug, output more messages
    debug_panda3d=False,  # debug panda3d
    debug_physics_world=False,  # only render physics world without model, a special debug option
    debug_static_world=False,  # debug static world
    log_level=logging.INFO,  # log level. logging.DEBUG/logging.CRITICAL or so on
    show_coordinates=False,  # show coordinates for maps and objects for debug

    # ===== GUI =====
    # Please see Documentation: GUI for more details
    # Whether to show these elements in the 3D scene
    show_fps=True,
    show_logo=True,
    show_mouse=True,
    show_skybox=True,
    show_terrain=True,
    show_interface=True,
    # Show marks for policies for debugging multi-policy setting
    show_policy_mark=False,
    # Show an arrow marks for providing navigation information
    show_interface_navi_mark=True,
    # A list showing sensor output on window. Its elements are chosen from sensors.keys() + "dashboard"
    interface_panel=["dashboard"],

    # ===== Record/Replay Metadata =====
    # Please see Documentation: Record and Replay for more details
    # When replay_episode is True, the episode metadata will be recorded
    record_episode=False,
    # The value should be None or the log data. If it is the later one, the simulator will replay logged scenario
    replay_episode=None,
    # When set to True, the replay system will only reconstruct the first frame from the logged scenario metadata
    only_reset_when_replay=False,
    # If True, when creating and replaying object trajectories, use the same ID as in dataset
    force_reuse_object_name=False,

    # ===== randomization =====
    num_scenarios=1  # the number of scenarios in this environment
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
        merged_config = self.default_config().update(config, False, ["agent_configs", "sensors"])
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

        # lazy initialization, create the main simulation in the lazy_init() func
        # self.engine: Optional[BaseEngine] = None

        # In MARL envs with respawn mechanism, varying episode lengths might happen.
        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)

        # press p to stop
        self.in_stop = False

        # scenarios
        self.start_index = 0

    def _post_process_config(self, config):
        """Add more special process to merged config"""
        # Cancel interface panel
        self.logger.info("Environment: {}".format(self.__class__.__name__))
        self.logger.info("MetaDrive version: {}".format(VERSION))
        if not config["show_interface"]:
            config["interface_panel"] = []

        # Adjust terrain
        n = config["map_region_size"]
        assert (n & (n - 1)) == 0 and 512 <= n <= 4096, "map_region_size should be pow of 2 and < 2048."
        TerrainProperty.map_region_size = config["map_region_size"]

        # Multi-Thread
        # if config["image_on_cuda"]:
        #     self.logger.info("Turn Off Multi-thread rendering due to image_on_cuda=True")
        #     config["multi_thread_render"] = False

        # Optimize sensor creation in none-screen mode
        if not config["use_render"] and not config["image_observation"]:
            filtered = {}
            for id, cfg in config["sensors"].items():
                if len(cfg) > 0 and not issubclass(cfg[0], BaseCamera) and id != "main_camera":
                    filtered[id] = cfg
            config["sensors"] = filtered
            config["interface_panel"] = []

        # Check sensor existence
        if config["use_render"] or "main_camera" in config["sensors"]:
            config["sensors"]["main_camera"] = ("MainCamera", *config["window_size"])

        # Merge dashboard config with sensors
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
                # Check whether some actions are not provided.
                given_keys = set(actions.keys())
                have_keys = set(self.agents.keys())
                assert given_keys == have_keys, "The input actions: {} have incompatible keys with existing {}!".format(
                    given_keys, have_keys
                )
            else:
                # That would be OK if extra actions is given. This is because, when evaluate a policy with naive
                # implementation, the "termination observation" will still be given in T=t-1. And at T=t, when you
                # collect action from policy(last_obs) without masking, then the action for "termination observation"
                # will still be computed. We just filter it out here.
                actions = {v_id: actions[v_id] for v_id in self.agents.keys()}
        return actions

    def _step_simulator(self, actions):
        # prepare for stepping the simulation
        scene_manager_before_step_infos = self.engine.before_step(actions)
        # step all entities and the simulator
        self.engine.step(self.config["decision_repeat"])
        # update states, if restore from episode data, position and heading will be force set in update_state() function
        scene_manager_after_step_infos = self.engine.after_step()

        # Note that we use shallow update for info dict in this function! This will accelerate system.
        return merge_dicts(
            scene_manager_after_step_infos, scene_manager_before_step_infos, allow_new_keys=True, without_copy=True
        )

    def reward_function(self, object_id: str) -> Tuple[float, Dict]:
        """
        Override this func to get a new reward function
        :param object_id: name of this object
        :return: reward, reward info
        """
        self.logger.warning("Reward function is not implemented. Return reward = 0", extra={"log_once": True})
        return 0, {}

    def cost_function(self, object_id: str) -> Tuple[float, Dict]:
        self.logger.warning("Cost function is not implemented. Return cost = 0", extra={"log_once": True})
        return 0, {}

    def done_function(self, object_id: str) -> Tuple[bool, Dict]:
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
        reset_info = self.engine.reset()
        self.reset_sensors()
        # render the scene
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
        """
        This is the developer API. Overriding it determines how to place sensors in the scene. You can mount it on an
        object or fix it at a given position for the whole episode.
        """
        # reset the cam at the start at the episode.
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
                        sensor.track(current_track_agent.origin, DEFAULT_SENSOR_OFFSET, DEFAULT_SENSOR_HPR)
        # Step the env to avoid the black screen in the first frame.
        self.engine.taskMgr.step()

    def _get_reset_return(self, reset_info):
        # TODO: figure out how to get the information of the before step
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
        """
        Wrap to single agent info
        """
        agent_info = data.pop(next(iter(self.agents.keys())))
        data.update(agent_info)
        return data

    def _get_step_return(self, actions, engine_info):
        # update obs, dones, rewards, costs, calculate done at first !
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

        # For extreme scenario only. Force to terminate all agents if the environmental step exceeds 5 times horizon.
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

    def for_each_agent(self, func, *args, **kwargs):
        return self.agent_manager.for_each_active_agents(func, *args, **kwargs)

    def get_single_observation(self):
        """
        Get the observation for one object
        """
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
    def num_scenarios(self):
        return self.config["num_scenarios"]

    @property
    def observations(self):
        """
        Return observations of active and controllable agents
        :return: Dict
        """
        return self.agent_manager.get_observations()

    @property
    def observation_space(self) -> gym.Space:
        """
        Return observation spaces of active and controllable agents
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
        Return action spaces of active and controllable agents. Generally, it is defined in AgentManager. But you can
        still overwrite this function to define the action space for the environment.
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
        self.logger.warning("env.vehicles will be deprecated soon. Use env.agents instead", extra={"log_once": True})
        return self.agents

    @property
    def vehicle(self):
        self.logger.warning("env.vehicle will be deprecated soon. Use env.agent instead", extra={"log_once": True})
        return self.agent

    @property
    def agents(self):
        """
        Return all active agents
        :return: Dict[agent_id:agent]
        """
        return self.agent_manager.active_agents

    @property
    def agent(self):
        """A helper to return the agent only in the single-agent environment!"""
        assert len(self.agents) == 1, (
            "env.agent is only supported in single-agent environment!"
            if len(self.agents) > 1 else "Please initialize the environment first!"
        )
        return self.agents[DEFAULT_AGENT]

    @property
    def agents_including_just_terminated(self):
        """
        Return all agents that occupy some space in current environments
        :return: Dict[agent_id:vehicle]
        """
        ret = self.agent_manager.active_agents
        ret.update(self.agent_manager.just_terminated_agents)
        return ret

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
                sensor.track(current_track_agent.origin, constants.DEFAULT_SENSOR_OFFSET, DEFAULT_SENSOR_HPR)
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
    while True:
        env.step(env.action_space.sample())
