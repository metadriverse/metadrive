import copy
import json
import logging
import os.path as osp
import sys
from typing import Union, Dict, AnyStr, Optional, Tuple

import numpy as np

from pgdrive.constants import DEFAULT_AGENT
from pgdrive.envs.base_env import BasePGDriveEnv
from pgdrive.obs.observation_type import LidarStateObservation, ImageStateObservation
from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.scene_creator.map import Map, MapGenerateMethod, parse_map_config, PGMap
from pgdrive.scene_creator.vehicle.base_vehicle import BaseVehicle
from pgdrive.scene_manager.traffic_manager import TrafficMode
from pgdrive.utils import clip, PGConfig, recursive_equal, get_np_random, concat_step_infos
from pgdrive.world.chase_camera import ChaseCamera
from pgdrive.world.manual_controller import KeyboardController, JoystickController
from pgdrive.world.pg_world import PGWorld

pregenerated_map_file = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "assets", "maps", "PGDrive-maps.json")

PGDriveEnvV1_DEFAULT_CONFIG = dict(
    # ===== Generalization =====
    start_seed=0,
    environment_num=1,

    # ===== Map Config =====
    map=3,  # int or string: an easy way to fill map_config
    map_config={
        Map.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        Map.GENERATE_CONFIG: None,  # it can be a file path / block num / block ID sequence
        Map.LANE_WIDTH: 3.5,
        Map.LANE_NUM: 3,
        Map.SEED: 10,
        "draw_map_resolution": 1024  # Drawing the map in a canvas of (x, x) pixels.
    },
    load_map_from_json=True,  # Whether to load maps from pre-generated file
    _load_map_from_json=pregenerated_map_file,  # The path to the pre-generated file

    # ==== agents config =====
    num_agents=1,

    # ===== Observation =====
    use_topdown=False,  # Use top-down view
    use_image=False,

    # ===== Traffic =====
    traffic_density=0.1,
    traffic_mode=TrafficMode.Trigger,  # "reborn", "trigger", "hybrid"
    random_traffic=False,  # Traffic is randomized at default.

    # ===== Object =====
    accident_prob=0.,  # accident may happen on each block with this probability, except multi-exits block

    # ===== Others =====
    auto_termination=True,  # Whether to done the environment after 250*(num_blocks+1) steps.

    # ===== Single-agent vehicle config =====
    vehicle_config=dict(
        # ===== vehicle module config =====
        lidar=dict(num_lasers=240, distance=50, num_others=4),  # laser num, distance, other vehicle info num
        show_lidar=False,
        mini_map=(84, 84, 250),  # buffer length, width
        rgb_cam=(84, 84),  # buffer length, width
        depth_cam=(84, 84, True),  # buffer length, width, view_ground
        show_navi_mark=True,
        increment_steering=False,
        wheel_friction=0.6,
        side_detector=dict(num_lasers=2, distance=50),  # laser num, distance
        show_side_detector=False,
        lane_line_detector=dict(num_lasers=2, distance=20),  # laser num, distance
        show_lane_line_detector=False,

        # ===== use image =====
        image_source="rgb_cam",  # take effect when only when use_image == True
        # use_image=False,
        # rgb_clip=True,

        # ===== vehicle born =====
        born_lane_index=(FirstBlock.NODE_1, FirstBlock.NODE_2, 0),
        born_longitude=5.0,
        born_lateral=0.0,

        # ==== others ====
        overtake_stat=False,  # we usually set to True when evaluation
        action_check=False,
        use_saver=False,
        save_level=0.5,
        use_lane_line_detector=False,
    ),
    rgb_clip=True,

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
    out_of_road_cost=1.
)


class PGDriveEnv(BasePGDriveEnv):
    @classmethod
    def default_config(cls) -> "PGConfig":
        config = super(PGDriveEnv, cls).default_config()
        config.update(PGDriveEnvV1_DEFAULT_CONFIG)
        config.register_type("map", str, int)
        config["map_config"].register_type("config", None)
        return config

    def __init__(self, config: dict = None):
        super(PGDriveEnv, self).__init__(config)

        self.current_track_vehicle: Optional[BaseVehicle] = None
        self.current_track_vehicle_id: Optional[str] = None

    def _process_config(self, config: Union[dict, "PGConfig"]) -> "PGConfig":
        """Check, update, sync and overwrite some config."""
        config["map_config"] = parse_map_config(
            easy_map_config=config["map"], new_map_config=config["map_config"], default_config=self.default_config_copy
        )
        config["pg_world_config"].update(
            {
                "use_render": config["use_render"],
                "use_image": config["use_image"],
                "debug": config["debug"],
                "decision_repeat": config["decision_repeat"],
                "fast_launch_window": config["fast"]
            }
        )
        config["vehicle_config"].update({"use_render": config["use_render"], "use_image": config["use_image"]})
        return config

    def _setup_pg_world(self) -> "PGWorld":
        pg_world = super(PGDriveEnv, self)._setup_pg_world()
        pg_world.accept("r", self.reset)
        pg_world.accept("escape", sys.exit)
        # Press t can let expert take over. But this function is still experimental.
        pg_world.accept("t", self.toggle_expert_takeover)
        # capture all figs
        pg_world.accept("p", self.capture)
        return pg_world

    def _after_lazy_init(self):
        if self.config["manual_control"]:
            if self.config["controller"] == "keyboard":
                self.controller = KeyboardController(pg_world=self.pg_world)
            elif self.config["controller"] == "joystick":
                self.controller = JoystickController()
            else:
                raise ValueError("No such a controller type: {}".format(self.config["controller"]))

        # initialize track vehicles
        # first tracked vehicles
        vehicles = sorted(self.vehicles.items())
        self.current_track_vehicle = vehicles[0][1]
        self.current_track_vehicle_id = vehicles[0][0]
        for _, vehicle in vehicles:
            if vehicle is not self.current_track_vehicle:
                # for display
                vehicle.remove_display_region()

        # for manual_control and main camera type
        if (self.config["use_render"] or self.config["use_image"]) and self.config["use_chase_camera"]:
            self.main_camera = ChaseCamera(self.pg_world.cam, self.config["camera_height"], 7, self.pg_world)
            self.main_camera.set_follow_lane(self.config["use_chase_camera_follow_lane"])
            self.main_camera.chase(self.current_track_vehicle, self.pg_world)
        self.pg_world.accept("n", self.chase_another_v)

        # for manual_control and main camera type
        if (self.config["use_render"] or self.config["use_image"]) and self.config["use_chase_camera"]:
            self.main_camera = ChaseCamera(self.pg_world.cam, self.config["camera_height"], 7, self.pg_world)
            self.main_camera.set_follow_lane(self.config["use_chase_camera_follow_lane"])
            self.main_camera.chase(self.current_track_vehicle, self.pg_world)
        self.pg_world.accept("n", self.chase_another_v)

    def _get_observations(self):
        return {self.DEFAULT_AGENT: self.get_single_observation(self.config["vehicle_config"])}

    def _preprocess_actions(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray]]) \
            -> Tuple[Union[np.ndarray, Dict[AnyStr, np.ndarray]], Dict]:

        if self.config["manual_control"] and self.config["use_render"] \
                and self.current_track_vehicle_id in self.vehicles.keys():
            action = self.controller.process_input()
            if self.num_agents == 1:
                actions = action
            else:
                actions[self.current_track_vehicle_id] = action

        if self.num_agents == 1:
            actions = {DEFAULT_AGENT: actions}

        # Check whether some actions are not provided.
        given_keys = set(actions.keys())
        have_keys = set(self.vehicles.keys())
        assert given_keys == have_keys, "The input actions: {} have incompatible keys with existing {}!".format(
            given_keys, have_keys
        )

        saver_info = dict()
        for v_id, v in self.vehicles.items():
            actions[v_id], saver_info[v_id] = self.saver(v_id, v, actions)
        return actions, saver_info

    def _get_step_return(self, actions, step_infos):
        """Don't need to copy anything here!"""
        # update obs, dones, rewards, costs, calculate done at first !
        obses = {}
        done_infos = {}
        cost_infos = {}
        reward_infos = {}
        rewards = {}
        for v_id, v in self.vehicles.items():
            obses[v_id] = self.observations[v_id].observe(v)
            done_function_result, done_infos[v_id] = self.done_function(v)
            rewards[v_id], reward_infos[v_id] = self.reward_function(v)
            _, cost_infos[v_id] = self.cost_function(v)
            self.dones[v_id] = done_function_result or self.dones[v_id]

        should_done = self.config["auto_termination"] and (self.episode_steps >= (self.current_map.num_blocks * 250))

        termination_infos = self.for_each_vehicle(_auto_termination, should_done)

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

        if self.num_agents == 1:
            return obses[DEFAULT_AGENT], rewards[DEFAULT_AGENT], self.dones[DEFAULT_AGENT], step_infos[DEFAULT_AGENT]
        else:
            return obses, rewards, self.dones, step_infos

    def done_function(self, vehicle):
        done = False
        done_info = dict(crash_vehicle=False, crash_object=False, out_of_road=False, arrive_dest=False)
        if vehicle.arrive_destination:
            done = True
            logging.info("Episode ended! Reason: arrive_dest.")
            done_info["arrive_dest"] = True
        elif vehicle.crash_vehicle:
            done = True
            logging.info("Episode ended! Reason: crash. ")
            done_info["crash_vehicle"] = True
        elif vehicle.out_of_route or not vehicle.on_lane or vehicle.crash_sidewalk:
            done = True
            logging.info("Episode ended! Reason: out_of_road.")
            done_info["out_of_road"] = True
        elif vehicle.crash_object:
            done = True
            done_info["crash_object"] = True

        # for compatibility
        # crash almost equals to crashing with vehicles
        done_info["crash"] = done_info["crash_vehicle"] or done_info["crash_object"]
        return done, done_info

    def cost_function(self, vehicle):
        step_info = dict()
        step_info["cost"] = 0
        if vehicle.crash_vehicle:
            step_info["cost"] = self.config["crash_vehicle_cost"]
        elif vehicle.crash_object:
            step_info["cost"] = self.config["crash_object_cost"]
        elif vehicle.out_of_route:
            step_info["cost"] = self.config["out_of_road_cost"]
        return step_info['cost'], step_info

    def reward_function(self, vehicle):
        """
        Override this func to get a new reward function
        :param vehicle: BaseVehicle
        :return: reward
        """
        step_info = dict()
        action = vehicle.last_current_action[1]
        # Reward for moving forward in current lane
        current_lane = vehicle.lane
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        reward = 0.0
        lateral_factor = clip(
            1 - 2 * abs(lateral_now) / vehicle.routing_localization.get_current_lane_width(), 0.0, 1.0
        )
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor

        # Penalty for frequent steering
        steering_change = abs(vehicle.last_current_action[0][0] - vehicle.last_current_action[1][0])
        steering_penalty = self.config["steering_penalty"] * steering_change * vehicle.speed / 20
        reward -= steering_penalty

        # Penalty for frequent acceleration / brake
        acceleration_penalty = self.config["acceleration_penalty"] * ((action[1])**2)
        reward -= acceleration_penalty

        # Penalty for waiting
        low_speed_penalty = 0
        if vehicle.speed < 1:
            low_speed_penalty = self.config["low_speed_penalty"]  # encourage car
        reward -= low_speed_penalty
        reward -= self.config["general_penalty"]

        reward += self.config["speed_reward"] * (vehicle.speed / vehicle.max_speed)
        step_info["step_reward"] = reward

        # for done
        if vehicle.crash_vehicle:
            reward -= self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward -= self.config["crash_object_penalty"]
        elif vehicle.out_of_route:
            reward -= self.config["out_of_road_penalty"]
        elif vehicle.arrive_destination:
            reward += self.config["success_reward"]

        return reward, step_info

    def _reset_vehicles(self):
        self.vehicles.update(self.done_vehicles)
        self.done_vehicles = {}
        self.for_each_vehicle(lambda v: v.reset(self.current_map))

    def _get_reset_return(self):
        ret = {}
        self.for_each_vehicle(lambda v: v.update_state())
        for v_id, v in self.vehicles.items():
            self.observations[v_id].reset(self, v)
            ret[v_id] = self.observations[v_id].observe(v)
        return ret[DEFAULT_AGENT] if self.num_agents == 1 else ret

    def _update_map(self, episode_data: dict = None):
        if episode_data is not None:
            # Since in episode data map data only contains one map, values()[0] is the map_parameters
            map_data = episode_data["map_data"].values()
            assert len(map_data) > 0, "Can not find map info in episode data"
            for map in map_data:
                blocks_info = map

            map_config = self.config["map_config"].copy()
            map_config[Map.GENERATE_TYPE] = MapGenerateMethod.PG_MAP_FILE
            map_config[Map.GENERATE_CONFIG] = blocks_info
            self.current_map = PGMap(self.pg_world, map_config)
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

            new_map = PGMap(self.pg_world, map_config)
            self.maps[self.current_seed] = new_map
            self.current_map = self.maps[self.current_seed]
        else:
            self.current_map = self.maps[self.current_seed]
            assert isinstance(self.current_map, Map), "Map should be an instance of Map() class"
            self.current_map.load_to_pg_world(self.pg_world)

    def dump_all_maps(self):
        assert self.pg_world is None, "We assume you generate map files in independent tasks (not in training). " \
                                      "So you should run the generating script without calling reset of the " \
                                      "environment."

        self.lazy_init()  # it only works the first time when reset() is called to avoid the error when render
        assert self.pg_world is not None
        self.pg_world.clear_world()

        for seed in range(self.start_seed, self.start_seed + self.env_num):
            print(seed)
            map_config = copy.deepcopy(self.config["map_config"])
            map_config.update({"seed": seed})
            new_map = PGMap(self.pg_world, map_config)
            self.maps[seed] = new_map
            new_map.unload_from_pg_world(self.pg_world)
            logging.info("Finish generating map with seed: {}".format(seed))

        map_data = dict()
        for seed, map in self.maps.items():
            assert map is not None
            map_data[seed] = map.save_map()

        return_data = dict(map_config=self.config["map_config"].copy().get_dict(), map_data=copy.deepcopy(map_data))
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
            map_config[Map.GENERATE_TYPE] = MapGenerateMethod.PG_MAP_FILE
            map_config[Map.GENERATE_CONFIG] = data["map_data"][str(seed)]
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
                    restored_data["map_config"], self.config["map_config"]
                )
            )
            self.config["load_map_from_json"] = False  # Don't fall into this function again.
            return False

    def toggle_expert_takeover(self):
        """
        Only take effect whene vehicle num==1
        :return: None
        """
        self.current_track_vehicle._expert_takeover = not self.current_track_vehicle._expert_takeover

    def chase_another_v(self) -> (str, BaseVehicle):
        vehicles = sorted(list(self.vehicles.items()) + list(self.done_vehicles.items())) * 2
        for index, v in enumerate(vehicles):
            if vehicles[index - 1][1] == self.current_track_vehicle:
                self.current_track_vehicle.remove_display_region()
                self.current_track_vehicle = v[1]
                self.current_track_vehicle_id = v[0]
                self.current_track_vehicle.add_to_display()
                self.main_camera.chase(self.current_track_vehicle, self.pg_world)
                return

    def saver(self, v_id: str, vehicle: BaseVehicle, actions):
        """
        Rule to enable saver
        :param v_id: id of a vehicle
        :param vehicle:BaseVehicle that need protection of saver
        :param actions: original actions of all vehicles
        :return: a new action to override original action
        """
        action = actions[v_id]
        steering = action[0]
        throttle = action[1]
        if vehicle.vehicle_config["use_saver"] or vehicle._expert_takeover:
            # saver can be used for human or another AI
            save_level = vehicle.vehicle_config["save_level"] if not vehicle._expert_takeover else 1.0
            obs = self.observations[v_id].observe(vehicle)
            from pgdrive.examples.ppo_expert import expert
            try:
                saver_a = expert(obs, deterministic=False)
            except ValueError:
                print("Expert can not takeover, due to observation space mismathing!")
                saver_a = action
            else:
                if save_level > 0.9:
                    steering = saver_a[0]
                    throttle = saver_a[1]
                elif save_level > 1e-3:
                    heading_diff = vehicle.heading_diff(vehicle.lane) - 0.5
                    f = min(1 + abs(heading_diff) * vehicle.speed * vehicle.max_speed, save_level * 10)
                    # for out of road
                    if (obs[0] < 0.04 * f and heading_diff < 0) or (obs[1] < 0.04 * f and heading_diff > 0) or obs[
                        0] <= 1e-3 or \
                            obs[
                                1] <= 1e-3:
                        steering = saver_a[0]
                        throttle = saver_a[1]
                        if vehicle.speed < 5:
                            throttle = 0.5
                    # if saver_a[1] * vehicle.speed < -40 and action[1] > 0:
                    #     throttle = saver_a[1]

                    # for collision
                    lidar_p = vehicle.lidar.get_cloud_points()
                    left = int(vehicle.lidar.num_lasers / 4)
                    right = int(vehicle.lidar.num_lasers / 4 * 3)
                    if min(lidar_p[left - 4:left + 6]) < (save_level + 0.1) / 10 or min(lidar_p[right - 4:right + 6]
                                                                                        ) < (save_level + 0.1) / 10:
                        # lateral safe distance 2.0m
                        steering = saver_a[0]
                    if action[1] >= 0 and saver_a[1] <= 0 and min(min(lidar_p[0:10]), min(lidar_p[-10:])) < save_level:
                        # longitude safe distance 15 m
                        throttle = saver_a[1]

        # indicate if current frame is takeover step
        pre_save = vehicle.takeover
        vehicle.takeover = True if action[0] != steering or action[1] != throttle else False
        saver_info = {
            "takeover_start": True if not pre_save and vehicle.takeover else False,
            "takeover_end": True if pre_save and not vehicle.takeover else False,
            "takeover": vehicle.takeover
        }
        return (steering, throttle), saver_info

    def get_single_observation(self, vehicle_config: "PGConfig") -> "ObservationType":
        if self.config["use_image"]:
            o = ImageStateObservation(vehicle_config)
        else:
            o = LidarStateObservation(vehicle_config)
        return o


def _auto_termination(vehicle, should_done):
    return {"max_step": True if should_done else False}


if __name__ == '__main__':

    def _act(env, action):
        assert env.action_space.contains(action)
        obs, reward, done, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isscalar(reward)
        assert isinstance(info, dict)

    env = PGDriveEnv()
    try:
        obs = env.reset()
        assert env.observation_space.contains(obs)
        _act(env, env.action_space.sample())
        for x in [-1, 0, 1]:
            env.reset()
            for y in [-1, 0, 1]:
                _act(env, [x, y])
    finally:
        env.close()
