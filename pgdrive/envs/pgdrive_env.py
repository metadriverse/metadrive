import copy
import json
import logging
import os.path as osp
import sys
import time
from typing import Union, Optional, Iterable

import gym
import numpy as np
from panda3d.core import PNMImage
from pgdrive.envs.observation_type import LidarStateObservation, ImageStateObservation
from pgdrive.pg_config import PGConfig
from pgdrive.scene_creator.ego_vehicle.base_vehicle import BaseVehicle
from pgdrive.scene_creator.ego_vehicle.vehicle_module.depth_camera import DepthCamera
from pgdrive.scene_creator.ego_vehicle.vehicle_module.mini_map import MiniMap
from pgdrive.scene_creator.ego_vehicle.vehicle_module.rgb_camera import RGBCamera
from pgdrive.scene_creator.map import Map, MapGenerateMethod, parse_map_config
from pgdrive.scene_manager.scene_manager import SceneManager
from pgdrive.scene_manager.traffic_manager import TrafficMode
from pgdrive.utils import recursive_equal, safe_clip, clip, get_np_random
from pgdrive.world import RENDER_MODE_NONE
from pgdrive.world.chase_camera import ChaseCamera
from pgdrive.world.manual_controller import KeyboardController, JoystickController
from pgdrive.world.pg_world import PGWorld

pregenerated_map_file = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "assets", "maps", "PGDrive-maps.json")


class PGDriveEnv(gym.Env):
    @staticmethod
    def default_config() -> PGConfig:
        env_config = dict(

            # ===== Rendering =====
            use_render=False,  # pop a window to render or not
            # force_fps=None,
            debug=False,
            cull_scene=True,  # only for debug use
            manual_control=False,
            controller="keyboard",  # "joystick" or "keyboard"
            use_chase_camera=True,
            camera_height=1.8,

            # ===== Traffic =====
            traffic_density=0.1,
            traffic_mode=TrafficMode.Trigger,
            random_traffic=False,  # Traffic is randomized at default.

            # ===== Object =====
            accident_prob=0.,  # accident may happen on each block with this probability, except multi-exits block

            # ===== Observation =====
            use_image=False,  # Use first view
            use_topdown=False,  # Use top-down view
            rgb_clip=True,
            vehicle_config=dict(),  # use default vehicle modules see more in BaseVehicle
            image_source="rgb_cam",  # mini_map or rgb_cam or depth cam

            # ===== Map Config =====
            map=3,  # int or string: an easy way to fill map_config
            map_config=dict(),
            load_map_from_json=True,  # Whether to load maps from pre-generated file
            _load_map_from_json=pregenerated_map_file,  # The path to the pre-generated file

            # ===== Generalization =====
            start_seed=0,
            environment_num=1,

            # ===== Action =====
            decision_repeat=5,

            # ===== Reward Scheme =====
            success_reward=20,
            out_of_road_penalty=5,
            crash_vehicle_penalty=10,
            crash_object_penalty=2,
            acceleration_penalty=0.0,
            steering_penalty=0.1,
            low_speed_penalty=0.0,
            driving_reward=1.0,
            general_penalty=0.0,
            speed_reward=0.1,

            # ===== Cost Scheme =====
            crash_vehicle_cost=1,
            crash_object_cost=1,
            out_of_road_cost=1.,

            # ===== Others =====
            pg_world_config=dict(),
            use_increment_steering=False,
            action_check=False,
            record_episode=False,
            use_saver=False,
            save_level=0.5,

            # ===== stat =====
            overtake_stat=False
        )
        config = PGConfig(env_config)
        config.register_type("map", str, int)
        return config

    def __init__(self, config: dict = None):
        self.config = self.default_config()
        if config:
            self.config.update(config)

        # set their value after vehicle created
        self.observation = self.initialize_observation()
        self.observation_space = self.observation.observation_space
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2, ), dtype=np.float32)

        self.start_seed = self.config["start_seed"]
        self.env_num = self.config["environment_num"]
        self.use_render = self.config["use_render"]

        # process map config
        self.config["map_config"] = parse_map_config(self.config["map"], self.config["map_config"])
        self.map_config = self.config["map_config"]

        pg_world_config = self.config["pg_world_config"]
        pg_world_config.update(
            {
                "use_render": self.use_render,
                "use_image": self.config["use_image"],
                # "use_topdown": self.config["use_topdown"],
                "debug": self.config["debug"],
                # "force_fps": self.config["force_fps"],
                "decision_repeat": self.config["decision_repeat"],
            }
        )
        self.pg_world_config = pg_world_config

        # lazy initialization, create the main vehicle in the lazy_init() func
        self.pg_world: Optional[PGWorld] = None
        self.scene_manager: Optional[SceneManager] = None
        self.main_camera = None
        self.controller = None
        self._expert_take_over = False
        self.restored_maps = dict()

        self.maps = {_seed: None for _seed in range(self.start_seed, self.start_seed + self.env_num)}
        self.current_seed = self.start_seed
        self.current_map = None
        self.vehicle = None  # Ego vehicle
        self.done = False
        self.takeover = False
        self.step_info = None
        self.front_vehicles = None
        self.back_vehicles = None

    def initialize_observation(self):
        vehicle_config = BaseVehicle.get_vehicle_config(self.config["vehicle_config"])
        if self.config["use_image"]:
            o = ImageStateObservation(vehicle_config, self.config["image_source"], self.config["rgb_clip"])
        else:
            o = LidarStateObservation(vehicle_config)
        return o

    def lazy_init(self):
        """
        Only init once in runtime, variable here exists till the close_env is called
        :return: None
        """
        # It is the true init() func to create the main vehicle and its module
        if self.pg_world is not None:
            return

        # init world
        self.pg_world = PGWorld(self.pg_world_config)
        self.pg_world.accept("r", self.reset)
        self.pg_world.accept("escape", sys.exit)

        # Press t can let expert take over. But this function is still experimental.
        self.pg_world.accept("t", self.toggle_expert_take_over)

        # capture all figs
        self.pg_world.accept("p", self.capture)

        # init traffic manager
        self.scene_manager = SceneManager(
            self.pg_world, self.config["traffic_mode"], self.config["random_traffic"], self.config["record_episode"],
            self.config["cull_scene"]
        )

        if self.config["manual_control"]:
            if self.config["controller"] == "keyboard":
                self.controller = KeyboardController(pg_world=self.pg_world)
            elif self.config["controller"] == "joystick":
                self.controller = JoystickController()
            else:
                raise ValueError("No such a controller type: {}".format(self.config["controller"]))

        # init vehicle
        v_config = self.config["vehicle_config"]
        self.vehicle = BaseVehicle(self.pg_world, v_config)

        # for manual_control and main camera type
        if (self.config["use_render"] or self.config["use_image"]) and self.config["use_chase_camera"]:
            self.main_camera = ChaseCamera(
                self.pg_world.cam, self.vehicle, self.config["camera_height"], 7, self.pg_world
            )
        # add sensors
        self.add_modules_for_vehicle()

    def step(self, action: np.ndarray):
        # add custom metric in info
        self.step_info = {"raw_action": (action[0], action[1]), "cost": 0}

        if self.config["action_check"]:
            assert self.action_space.contains(action), "Input {} is not compatible with action space {}!".format(
                action, self.action_space
            )

        if self.config["manual_control"] and self.use_render:
            action = self.controller.process_input()
            action = self.expert_take_over(action)

        # filter by saver to protect
        action = self.saver(action)

        # protect agent from nan error
        action = safe_clip(action, min_val=self.action_space.low[0], max_val=self.action_space.high[0])

        # preprocess
        self.scene_manager.prepare_step(action)

        # step all entities
        self.scene_manager.step(self.config["decision_repeat"])

        # update states, if restore from episode data, position and heading will be force set in update_state() function
        done = self.scene_manager.update_state()

        # update obs
        obs = self.observation.observe(self.vehicle)

        # update rl info
        self.done = self.done or done
        step_reward = self.reward(action)
        done_reward = self._done_episode()

        if self.done:
            step_reward = 0

        # update info
        self.step_info.update(
            {
                "velocity": float(self.vehicle.speed),
                "steering": float(self.vehicle.steering),
                "acceleration": float(self.vehicle.throttle_brake),
                "step_reward": float(step_reward),
                "takeover": self.takeover,
            }
        )
        self.custom_info_callback()

        return obs, step_reward + done_reward, self.done, self.step_info

    def render(self, mode='human', text: Optional[Union[dict, str]] = None) -> Optional[np.ndarray]:
        """
        This is a pseudo-render function, only used to update onscreen message when using panda3d backend
        :param mode: 'rgb'/'human'
        :param text:text to show
        :return: when mode is 'rgb', image array is returned
        """
        assert self.use_render or self.pg_world.mode != RENDER_MODE_NONE, ("render is off now, can not render")
        self.pg_world.render_frame(text)
        if mode != "human" and self.config["use_image"]:
            # fetch img from img stack to be make this func compatible with other render func in RL setting
            return self.observation.img_obs.get_image()

        if mode == "rgb_array" and self.config["use_render"]:
            if not hasattr(self, "_temporary_img_obs"):
                from pgdrive.envs.observation_type import ImageObservation
                image_source = "rgb_cam"
                self.temporary_img_obs = ImageObservation(self.vehicle.vehicle_config, image_source, False)
            self.temporary_img_obs.observe(self.vehicle.image_sensors[image_source])
            return self.temporary_img_obs.get_image()

        # logging.warning("You do not set 'use_image' or 'use_image' to True, so no image will be returned!")
        return None

    def reset(self, episode_data: dict = None):
        """
        Reset the env, scene can be restored and replayed by giving episode_data
        Reset the environment or load an episode from episode data to recover is
        :param episode_data: Feed the episode data to replay an episode
        :return: None
        """
        self.lazy_init()  # it only works the first time when reset() is called to avoid the error when render
        self.done = False
        self.takeover = False

        # clear world and traffic manager
        self.pg_world.clear_world()

        # select_map
        self.update_map(episode_data)

        # reset main vehicle
        self.vehicle.reset(self.current_map, self.vehicle.born_place, 0.0)

        # generate new traffic according to the map
        self.scene_manager.reset(
            self.current_map,
            self.vehicle,
            self.config["traffic_density"],
            self.config["accident_prob"],
            episode_data=episode_data
        )

        self.front_vehicles = set()
        self.back_vehicles = set()
        return self._get_reset_return()

    def _get_reset_return(self):
        self.vehicle.prepare_step(np.array([0.0, 0.0]))
        self.vehicle.update_state()

        self.observation.reset(self)

        o = self.observation.observe(self.vehicle)
        return o

    def reward(self, action):
        """
        Override this func to get a new reward function
        :param action: [steering, throttle/brake]
        :return: reward
        """
        # Reward for moving forward in current lane
        current_lane = self.vehicle.lane
        long_last, _ = current_lane.local_coordinates(self.vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(self.vehicle.position)

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        reward = 0.0
        lateral_factor = clip(1 - 2 * abs(lateral_now) / self.current_map.lane_width, 0.0, 1.0)
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor

        # Penalty for frequent steering
        steering_change = abs(self.vehicle.last_current_action[0][0] - self.vehicle.last_current_action[1][0])
        steering_penalty = self.config["steering_penalty"] * steering_change * self.vehicle.speed / 20
        reward -= steering_penalty

        # Penalty for frequent acceleration / brake
        acceleration_penalty = self.config["acceleration_penalty"] * ((action[1])**2)
        reward -= acceleration_penalty

        # Penalty for waiting
        low_speed_penalty = 0
        if self.vehicle.speed < 1:
            low_speed_penalty = self.config["low_speed_penalty"]  # encourage car
        reward -= low_speed_penalty
        reward -= self.config["general_penalty"]
        reward += self.config["speed_reward"] * (self.vehicle.speed / self.vehicle.max_speed)

        return reward

    def _done_episode(self) -> (float, dict):
        reward_ = 0
        done_info = dict(crash_vehicle=False, crash_object=False, out_of_road=False, arrive_dest=False)
        long, lat = self.vehicle.routing_localization.final_lane.local_coordinates(self.vehicle.position)

        if self.vehicle.routing_localization.final_lane.length - 5 < long < self.vehicle.routing_localization.final_lane.length + 5 \
                and self.current_map.lane_width / 2 >= lat >= (
                0.5 - self.current_map.lane_num) * self.current_map.lane_width:
            self.done = True
            reward_ += self.config["success_reward"]
            logging.info("Episode ended! Reason: arrive_dest.")
            done_info["arrive_dest"] = True
        elif self.vehicle.crash_vehicle:
            self.done = True
            reward_ -= self.config["crash_vehicle_penalty"]
            logging.info("Episode ended! Reason: crash vehicle")
            done_info["crash_vehicle"] = True
        elif self.vehicle.out_of_route or not self.vehicle.on_lane or self.vehicle.crash_side_walk:
            self.done = True
            reward_ -= self.config["out_of_road_penalty"]
            logging.info("Episode ended! Reason: out_of_road.")
            done_info["out_of_road"] = True
        elif self.vehicle.crash_object:
            self.done = True
            reward_ -= self.config["crash_object_penalty"]
            done_info["crash_object"] = True

        self.step_info.update(done_info)

        # ===== for old version compatibility =====
        # crash almost equals to crashing with vehicles
        self.step_info["crash"] = self.step_info["crash_vehicle"] or self.step_info["crash_object"]
        return reward_

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

            self.vehicle.destroy(self.pg_world)
            del self.vehicle
            self.vehicle = None

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

    def custom_info_callback(self):
        """
        Override it to add custom infomation
        :return: None
        """
        if self.config["overtake_stat"]:
            # use it only when evaluation
            self._overtake_stat()
        self.step_info["overtake_vehicle_num"] = len(self.front_vehicles.intersection(self.back_vehicles))

    def _overtake_stat(self):
        surrounding_vs = self.vehicle.lidar.get_surrounding_vehicles()
        routing = self.vehicle.routing_localization
        ckpt_idx = routing.target_checkpoints_index
        for surrounding_v in surrounding_vs:
            if surrounding_v.lane_index[:-1] == (routing.checkpoints[ckpt_idx[0]], routing.checkpoints[ckpt_idx[1]]):
                if self.vehicle.lane.local_coordinates(self.vehicle.position)[0] - \
                        self.vehicle.lane.local_coordinates(surrounding_v.position)[0] < 0:
                    self.front_vehicles.add(surrounding_v)
                    if surrounding_v in self.back_vehicles:
                        self.back_vehicles.remove(surrounding_v)
                else:
                    self.back_vehicles.add(surrounding_v)

    def update_map(self, episode_data: dict = None):
        if episode_data is not None:
            # Since in episode data map data only contains one map, values()[0] is the map_parameters
            map_data = episode_data["map_data"].values()
            assert len(map_data) > 0, "Can not find map info in episode data"
            for map in map_data:
                blocks_info = map
            map_config = {}
            map_config[Map.GENERATE_METHOD] = MapGenerateMethod.PG_MAP_FILE
            map_config[Map.GENERATE_PARA] = blocks_info
            self.current_map = Map(self.pg_world, map_config)
            return

        if self.config["load_map_from_json"] and self.current_map is None:
            assert self.config["_load_map_from_json"]
            self.load_all_maps_from_json(self.config["_load_map_from_json"])

        # remove map from world before adding
        if self.current_map is not None:
            self.current_map.unload_from_pg_world(self.pg_world)

        # create map
        self.current_seed = get_np_random().randint(self.start_seed, self.start_seed + self.env_num)
        if self.maps.get(self.current_seed, None) is None:

            if self.config["load_map_from_json"]:
                map_config = self.restored_maps.get(self.current_seed, None)
                assert map_config is not None
            else:
                map_config = self.config["map_config"]
                map_config.update({"seed": self.current_seed})

            new_map = Map(self.pg_world, map_config)
            self.maps[self.current_seed] = new_map
            self.current_map = self.maps[self.current_seed]
        else:
            self.current_map = self.maps[self.current_seed]
            assert isinstance(self.current_map, Map), "map should be an instance of Map() class"
            self.current_map.load_to_pg_world(self.pg_world)

    def add_modules_for_vehicle(self):
        # add vehicle module for training according to config
        vehicle_config = self.vehicle.vehicle_config
        self.vehicle.add_routing_localization(vehicle_config["show_navi_mark"])  # default added
        if not self.config["use_image"]:
            if vehicle_config["lidar"]["num_lasers"] > 0 and vehicle_config["lidar"]["distance"] > 0:
                self.vehicle.add_lidar(
                    num_lasers=vehicle_config["lidar"]["num_lasers"],
                    distance=vehicle_config["lidar"]["distance"],
                    show_lidar_point=vehicle_config["show_lidar"]
                )
            else:
                import logging
                logging.warning(
                    "You have set the lidar config to: {}, which seems to be invalid!".format(vehicle_config["lidar"])
                )

            if self.config["use_render"]:
                rgb_cam_config = vehicle_config["rgb_cam"]
                rgb_cam = RGBCamera(rgb_cam_config[0], rgb_cam_config[1], self.vehicle.chassis_np, self.pg_world)
                self.vehicle.add_image_sensor("rgb_cam", rgb_cam)

                mini_map = MiniMap(vehicle_config["mini_map"], self.vehicle.chassis_np, self.pg_world)
                self.vehicle.add_image_sensor("mini_map", mini_map)
            return

        if self.config["use_image"]:
            # 3 types image observation
            if self.config["image_source"] == "rgb_cam":
                rgb_cam_config = vehicle_config["rgb_cam"]
                rgb_cam = RGBCamera(rgb_cam_config[0], rgb_cam_config[1], self.vehicle.chassis_np, self.pg_world)
                self.vehicle.add_image_sensor("rgb_cam", rgb_cam)
            elif self.config["image_source"] == "mini_map":
                mini_map = MiniMap(vehicle_config["mini_map"], self.vehicle.chassis_np, self.pg_world)
                self.vehicle.add_image_sensor("mini_map", mini_map)
            elif self.config["image_source"] == "depth_cam":
                cam_config = vehicle_config["depth_cam"]
                depth_cam = DepthCamera(*cam_config, self.vehicle.chassis_np, self.pg_world)
                self.vehicle.add_image_sensor("depth_cam", depth_cam)
            else:
                raise ValueError("No module named {}".format(self.config["image_source"]))

        # load more sensors for visualization when render, only for beauty...
        if self.config["use_render"]:
            if self.config["image_source"] == "mini_map":
                rgb_cam_config = vehicle_config["rgb_cam"]
                rgb_cam = RGBCamera(rgb_cam_config[0], rgb_cam_config[1], self.vehicle.chassis_np, self.pg_world)
                self.vehicle.add_image_sensor("rgb_cam", rgb_cam)
            else:
                mini_map = MiniMap(vehicle_config["mini_map"], self.vehicle.chassis_np, self.pg_world)
                self.vehicle.add_image_sensor("mini_map", mini_map)

    def dump_all_maps(self):
        assert self.pg_world is None, "We assume you generate map files in independent tasks (not in training). " \
                                      "So you should run the generating script without calling reset of the " \
                                      "environment."

        self.lazy_init()  # it only works the first time when reset() is called to avoid the error when render
        self.pg_world.clear_world()

        for seed in range(self.start_seed, self.start_seed + self.env_num):
            print(seed)
            map_config = copy.deepcopy(self.config["map_config"])
            map_config.update({"seed": seed})
            new_map = Map(self.pg_world, map_config)
            self.maps[seed] = new_map
            new_map.unload_from_pg_world(self.pg_world)
            logging.info("Finish generating map with seed: {}".format(seed))

        map_data = dict()
        for seed, map in self.maps.items():
            assert map is not None
            map_data[seed] = map.save_map()

        return_data = dict(map_config=copy.deepcopy(self.config["map_config"]), map_data=copy.deepcopy(map_data))
        return return_data

    def load_all_maps(self, data):
        assert isinstance(data, dict)
        assert set(data.keys()) == set(["map_config", "map_data"])
        assert set(self.maps.keys()).issubset(set([int(v) for v in data["map_data"].keys()]))

        logging.info(
            "Restoring the maps from pre-generated file! "
            "We have {} maps in the file and restoring {} maps range from {} to {}".format(
                len(data["map_data"]), len(self.maps.keys()), min(self.maps.keys()), max(self.maps.keys())
            )
        )

        maps_collection_config = data["map_config"]
        assert set(self.config["map_config"].keys()) == set(maps_collection_config.keys())
        for k in self.config["map_config"]:
            assert maps_collection_config[k] == self.config["map_config"][k]

        # for seed, map_dict in data["map_data"].items():
        for seed in self.maps.keys():
            assert str(seed) in data["map_data"]
            assert self.maps[seed] is None
            map_config = {}
            map_config[Map.GENERATE_METHOD] = MapGenerateMethod.PG_MAP_FILE
            map_config[Map.GENERATE_PARA] = data["map_data"][str(seed)]
            self.restored_maps[seed] = map_config

    def load_all_maps_from_json(self, path):
        assert path.endswith(".json")
        assert osp.isfile(path)
        with open(path, "r") as f:
            restored_data = json.load(f)
        if recursive_equal(self.config["map_config"], restored_data["map_config"]) and \
                self.start_seed + self.env_num < len(restored_data["map_data"]):
            self.load_all_maps(restored_data)
            return True
        else:
            logging.warning(
                "Warning: The pre-generated maps is with config {}, but current environment's map "
                "config is {}.\nWe now fallback to BIG algorithm to generate map online!".format(
                    restored_data["map_config"], self.map_config
                )
            )
            self.config["load_map_from_json"] = False  # Don't fall into this function again.
            return False

    def force_close(self):
        print("Closing environment ... Please wait")
        self.close()
        time.sleep(2)  # Sleep two seconds
        raise KeyboardInterrupt("'Esc' is pressed. PGDrive exits now.")

    def set_current_seed(self, seed):
        self.current_seed = seed

    def get_map(self, resolution: Iterable = (512, 512)):
        return self.current_map.get_map_image_array(resolution)

    def get_vehicle_num(self):
        if self.scene_manager is None:
            return 0
        return self.scene_manager.get_vehicle_num()

    def expert_take_over(self, action):
        if self._expert_take_over:
            from pgdrive.examples.ppo_expert import expert
            return expert(self.observation.observe(self.vehicle))
        else:
            return action

    def saver(self, action):
        """
        Rule to enable saver
        :param action: original action
        :return: a new action to override original action
        """
        steering = action[0]
        throttle = action[1]
        if self.config["use_saver"] and not self._expert_take_over:
            # saver can be used for human or another AI
            save_level = self.config["save_level"]
            obs = self.observation.observe(self.vehicle)
            from pgdrive.examples.ppo_expert import expert
            saver_a = expert(obs, deterministic=False)
            if save_level > 0.9:
                steering = saver_a[0]
                throttle = saver_a[1]
            elif save_level > 1e-3:
                heading_diff = self.vehicle.heading_diff(self.vehicle.lane) - 0.5
                f = min(1 + abs(heading_diff) * self.vehicle.speed * self.vehicle.max_speed, save_level * 10)
                # for out of road
                if (obs[0] < 0.04 * f and heading_diff < 0) or (obs[1] < 0.04 * f and heading_diff > 0) or obs[
                    0] <= 1e-3 or \
                        obs[
                            1] <= 1e-3:
                    steering = saver_a[0]
                    throttle = saver_a[1]
                    if self.vehicle.speed < 5:
                        throttle = 0.5
                # if saver_a[1] * self.vehicle.speed < -40 and action[1] > 0:
                #     throttle = saver_a[1]

                # for collision
                lidar_p = self.vehicle.lidar.get_cloud_points()
                left = int(self.vehicle.lidar.num_lasers / 4)
                right = int(self.vehicle.lidar.num_lasers / 4 * 3)
                if min(lidar_p[left - 4:left + 6]) < (save_level + 0.1) / 10 or min(lidar_p[right - 4:right + 6]
                                                                                    ) < (save_level + 0.1) / 10:
                    # lateral safe distance 2.0m
                    steering = saver_a[0]
                if action[1] >= 0 and saver_a[1] <= 0 and min(min(lidar_p[0:10]), min(lidar_p[-10:])) < save_level:
                    # longitude safe distance 15 m
                    throttle = saver_a[1]

        # indicate if current frame is takeover step
        pre_save = self.takeover
        self.takeover = True if action[0] != steering or action[1] != throttle else False
        self.step_info["takeover_start"] = True if not pre_save and self.takeover else False
        self.step_info["takeover_end"] = True if pre_save and not self.takeover else False
        return steering, throttle

    def toggle_expert_take_over(self):
        self._expert_take_over = not self._expert_take_over

    def capture(self):
        img = PNMImage()
        self.pg_world.win.getScreenshot(img)
        img.write("main.jpg")

        for name, sensor in self.vehicle.image_sensors.items():
            if name == "mini_map":
                name = "lidar"
            sensor.save_image("{}.jpg".format(name))
        # if self.pg_world.highway_render is not None:
        #     self.pg_world.highway_render.get_screenshot("top_down.jpg")
