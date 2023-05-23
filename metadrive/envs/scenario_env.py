"""
This environment can load all scenarios exported from other environments via env.export_scenarios()
"""
import logging

import numpy as np

from metadrive.component.vehicle_module.vehicle_panel import VehiclePanel
from metadrive.component.vehicle_navigation_module.trajectory_navigation import TrajectoryNavigation
from metadrive.constants import TerminationState
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.base_env import BaseEnv
from metadrive.manager.scenario_curriculum_manager import ScenarioCurriculumManager
from metadrive.manager.scenario_data_manager import ScenarioDataManager
from metadrive.manager.scenario_light_manager import ScenarioLightManager
from metadrive.manager.scenario_map_manager import ScenarioMapManager
from metadrive.manager.waymo_traffic_manager import WaymoTrafficManager
from metadrive.obs.real_env_observation import ScenarioObservation
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.utils import clip
from metadrive.utils import get_np_random
from metadrive.utils.math import norm

SCENARIO_ENV_CONFIG = dict(
    # ===== Scenario Config =====
    data_directory=AssetLoader.file_path("waymo", return_raw_style=False),
    start_scenario_index=0,
    num_scenarios=3,
    sequential_seed=False,  # Whether to set seed (the index of map) sequentially across episodes
    worker_index=0,  # Allowing multi-worker sampling with Rllib
    num_workers=1,  # Allowing multi-worker sampling with Rllib

    # ===== Curriculum Config =====
    curriculum_level=1,  # i.e. set to 5 to split the data into 5 difficulty level
    episodes_to_evaluate_curriculum=None,
    target_success_rate=0.85,

    # ===== Map Config =====
    store_map=True,
    need_lane_localization=True,

    # ===== Traffic =====
    no_traffic=False,  # nothing will be generated including objects/pedestrian/vehicles
    no_static_vehicles=False,  # static vehicle will be removed
    no_light=False,  # no traffic light
    reactive_traffic=False,  # turn on to enable idm traffic
    filter_overlapping_car=True,  # If in one frame a traffic vehicle collides with ego car, it won't be created.
    even_sample_vehicle_class=True,  # to make the scene more diverse
    default_vehicle_in_traffic=False,

    # ===== Agent config =====
    vehicle_config=dict(
        lidar=dict(num_lasers=120, distance=50),
        lane_line_detector=dict(num_lasers=0, distance=50),
        side_detector=dict(num_lasers=12, distance=50),
        show_dest_mark=True,
        navigation_module=TrajectoryNavigation,
    ),

    # ===== Reward Scheme =====
    # See: https://github.com/metadriverse/metadrive/issues/283
    success_reward=10.0,
    out_of_road_penalty=10.0,
    crash_vehicle_penalty=1,
    crash_object_penalty=1.0,
    driving_reward=1.0,
    speed_reward=0.1,
    use_lateral_reward=False,

    # ===== Cost Scheme =====
    crash_vehicle_cost=1.0,
    crash_object_cost=1.0,
    out_of_road_cost=1.0,
    crash_human_cost=1.0,

    # ===== Termination Scheme =====
    out_of_route_done=False,
    crash_vehicle_done=False,
    relax_out_of_road_done=True,

    # ===== others =====
    interface_panel=[VehiclePanel],  # for boosting efficiency
    horizon=None,
)


class ScenarioEnv(BaseEnv):
    @classmethod
    def default_config(cls):
        config = super(ScenarioEnv, cls).default_config()
        config.update(SCENARIO_ENV_CONFIG)
        return config

    def __init__(self, config=None):
        super(ScenarioEnv, self).__init__(config)
        if self.config["curriculum_level"] > 1:
            assert self.config["num_scenarios"] % self.config["curriculum_level"] == 0, \
                "Each level should have the same number of scenarios"
            if self.config["num_workers"] > 1:
                num = int(self.config["num_scenarios"] / self.config["curriculum_level"])
                assert num % self.config["num_workers"] == 0
        if self.config["num_workers"] > 1:
            assert self.config["sequential_seed"], \
                "If using > 1 workers, you have to allow sequential_seed for consistency!"

    def _merge_extra_config(self, config):
        # config = self.default_config().update(config, allow_add_new_key=True)
        config = self.default_config().update(config, allow_add_new_key=False)
        return config

    def _get_observations(self):
        return {self.DEFAULT_AGENT: self.get_single_observation(self.config["vehicle_config"])}

    def get_single_observation(self, vehicle_config):
        o = ScenarioObservation(vehicle_config)
        return o

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

    def setup_engine(self):
        self.in_stop = False
        super(ScenarioEnv, self).setup_engine()
        self.engine.register_manager("data_manager", ScenarioDataManager())
        self.engine.register_manager("map_manager", ScenarioMapManager())
        if not self.config["no_traffic"]:
            self.engine.register_manager("traffic_manager", WaymoTrafficManager())
        if not self.config["no_light"]:
            self.engine.register_manager("light_manager", ScenarioLightManager())
        self.engine.register_manager("curriculum_manager", ScenarioCurriculumManager())
        self.engine.accept("p", self.stop)
        self.engine.accept("q", self.switch_to_third_person_view)
        self.engine.accept("b", self.switch_to_top_down_view)
        self.engine.accept("]", self.next_seed_reset)
        self.engine.accept("[", self.last_seed_reset)

    def next_seed_reset(self):
        if self.current_seed + 1 < self.config["start_scenario_index"] + self.config["num_scenarios"]:
            self.reset(self.current_seed + 1)
        else:
            logging.warning("Can't load next scenario! current seed is already the max scenario index")

    def last_seed_reset(self):
        if self.current_seed - 1 >= self.config["start_scenario_index"]:
            self.reset(self.current_seed - 1)
        else:
            logging.warning("Can't load last scenario! current seed is already the min scenario index")

    def step(self, actions):
        ret = super(ScenarioEnv, self).step(actions)
        while self.in_stop:
            self.engine.taskMgr.step()
        return ret

    def done_function(self, vehicle_id: str):
        vehicle = self.vehicles[vehicle_id]
        done = False
        done_info = dict(
            crash_vehicle=False, crash_object=False, crash_building=False, out_of_road=False, arrive_dest=False,
            max_step=False,
        )

        route_completion = vehicle.navigation.route_completion

        if self._is_arrive_destination(vehicle):
            done = True
            logging.info("Episode ended! Reason: arrive_dest.")
            done_info[TerminationState.SUCCESS] = True

            # log data to curriculum manager
            self.engine.curriculum_manager.log_episode(True, route_completion)

        elif self._is_out_of_road(vehicle) or route_completion < -0.1:
            done = True
            logging.info("Episode ended! Reason: out_of_road.")
            done_info[TerminationState.OUT_OF_ROAD] = True
        elif vehicle.crash_vehicle and self.config["crash_vehicle_done"]:
            done = True
            logging.info("Episode ended! Reason: crash vehicle ")
            done_info[TerminationState.CRASH_VEHICLE] = True
        elif vehicle.crash_object:
            done = True
            done_info[TerminationState.CRASH_OBJECT] = True
            logging.info("Episode ended! Reason: crash object ")
        elif vehicle.crash_building:
            done = True
            done_info[TerminationState.CRASH_BUILDING] = True
            logging.info("Episode ended! Reason: crash building ")

        elif self.config["horizon"] is not None and \
                self.episode_lengths[vehicle_id] >= self.config["horizon"] and not self.is_multi_agent:
            done = True
            done_info[TerminationState.MAX_STEP] = True
            logging.info("Episode ended! Reason: max step ")

        # for compatibility
        # crash almost equals to crashing with vehicles
        done_info[TerminationState.CRASH] = (
                done_info[TerminationState.CRASH_VEHICLE] or done_info[TerminationState.CRASH_OBJECT]
                or done_info[TerminationState.CRASH_BUILDING]
        )

        return done, done_info

    def cost_function(self, vehicle_id: str):
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()
        step_info["cost"] = 0
        if self._is_out_of_road(vehicle):
            step_info["cost"] = self.config["out_of_road_cost"]
        elif vehicle.crash_vehicle:
            step_info["cost"] = self.config["crash_vehicle_cost"]
        elif vehicle.crash_object:
            step_info["cost"] = self.config["crash_object_cost"]
        elif vehicle.crash_human:
            step_info["cost"] = self.config["crash_human_cost"]
        return step_info['cost'], step_info

    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
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
            self.engine.map_manager.current_sdc_route.local_coordinates(vehicle.position)[-1]

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral_reward"]:
            lateral_factor = clip(1 - 2 * abs(lateral_now) / 6, 0.0, 1.0)
        else:
            lateral_factor = 1.0

        # dense reward
        reward = 0
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor
        if vehicle.on_yellow_continuous_line or vehicle.crash_sidewalk:
            reward -= 1.0
        step_info["step_reward"] = reward

        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]

        # TODO LQY: all a callback to process these keys
        step_info["track_length"] = vehicle.navigation.reference_trajectory.length
        step_info["carsize"] = [vehicle.WIDTH, vehicle.LENGTH]
        # add some new and informative keys
        step_info["route_completion"] = vehicle.navigation.route_completion
        step_info["curriculum_level"] = self.engine.current_level
        step_info["scenario_index"] = self.engine.current_seed
        step_info["num_stored_maps"] = self.engine.map_manager.num_stored_maps
        step_info["scenario_difficulty"] = self.engine.data_manager.current_scenario_difficulty
        step_info["data_coverage"] = self.engine.data_manager.data_coverage
        step_info["curriculum_success"] = self.engine.curriculum_manager.current_success_rate
        step_info["curriculum_route_completion"] = self.engine.curriculum_manager.current_route_completion

        # Compute state difference metrics
        data = self.engine.data_manager.current_scenario
        agent_xy = vehicle.position
        if vehicle_id == "sdc" or vehicle_id == "default_agent":
            native_vid = data[ScenarioDescription.METADATA][ScenarioDescription.SDC_ID]
        else:
            native_vid = vehicle_id

        if native_vid in data["tracks"] and len(data["tracks"][native_vid]) > 0:
            expert_state_list = data["tracks"][native_vid]["state"]

            mask = expert_state_list["valid"]
            largest_valid_index = np.max(np.where(mask == True)[0])

            if self.episode_step > largest_valid_index:
                current_step = largest_valid_index
            else:
                current_step = self.episode_step

            while mask[current_step] == 0.0:
                current_step -= 1
                if current_step == 0:
                    break

            expert_xy = expert_state_list["position"][current_step][:2]
            diff = agent_xy - expert_xy
            dist = norm(diff[0], diff[1])
            step_info["distance_error"] = dist

            last_state = expert_state_list["position"][largest_valid_index]
            last_expert_xy = last_state[:2]
            diff = agent_xy - last_expert_xy
            last_dist = norm(diff[0], diff[1])
            step_info["distance_error_final"] = last_dist

            # reward = reward - self.config["distance_penalty"] * dist

        if hasattr(vehicle, "_dynamics_mode"):
            step_info["dynamics_mode"] = vehicle._dynamics_mode

        return reward, step_info

    def _is_arrive_destination(self, vehicle):
        # Use RC as the only criterion to determine arrival in Scenario env.
        route_completion = vehicle.navigation.route_completion
        if route_completion > 0.95 or vehicle.navigation.reference_trajectory.length < 2:
            # Route Completion ~= 1.0 or vehicle is static!
            return True
        else:
            return False

    def _is_out_of_road(self, vehicle):
        # A specified function to determine whether this vehicle should be done.

        if self.config["relax_out_of_road_done"]:
            # We prefer using this out of road termination criterion.
            agent_name = self.agent_manager.object_to_agent(vehicle.name)
            lat = abs(self.observations[agent_name].lateral_dist)
            done = lat > 5
            done = done
            return done

        done = vehicle.crash_sidewalk or vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line
        if self.config["out_of_route_done"]:
            agent_name = self.agent_manager.object_to_agent(vehicle.name)
            done = done or abs(self.observations[agent_name].lateral_dist) > 10
        return done

    def _reset_global_seed(self, force_seed=None):
        if force_seed is not None:
            current_seed = force_seed
        elif self.config["sequential_seed"]:
            current_seed = self.engine.global_seed
            if current_seed is None:
                current_seed = int(self.config["start_scenario_index"]) + int(self.config["worker_index"])
            else:
                current_seed += int(self.config["num_workers"])
            if current_seed >= self.config["start_scenario_index"] + int(self.config["num_scenarios"]):
                current_seed = int(self.config["start_scenario_index"]) + int(self.config["worker_index"])
        else:
            current_seed = get_np_random(None).randint(
                self.config["start_scenario_index"],
                self.config["start_scenario_index"] + int(self.config["num_scenarios"])
            )

        assert self.config["start_scenario_index"] <= current_seed < \
               self.config["start_scenario_index"] + self.config[
                   "num_scenarios"], "Force seed {} is out of range [{}, {}).".format(current_seed, self.config[
            "start_scenario_index"], self.config["start_scenario_index"] + self.config["num_scenarios"])
        self.seed(current_seed)

    def stop(self):
        self.in_stop = not self.in_stop


if __name__ == "__main__":
    env = ScenarioEnv(
        {
            "use_render": True,
            "agent_policy": ReplayEgoCarPolicy,
            "manual_control": False,
            "show_interface": True,
            "show_logo": False,
            "show_fps": False,
            # "debug": True,
            # "debug_static_world": True,
            # "no_traffic": True,
            # "no_light": True,
            # "debug":True,
            # "no_traffic":True,
            # "start_scenario_index": 192,
            # "start_scenario_index": 1000,
            "num_scenarios": 30,
            # "force_reuse_object_name": True,
            # "data_directory": "/home/shady/Downloads/test_processed",
            "horizon": 1000,
            "no_static_vehicles": True,
            # "show_policy_mark": True,
            # "show_coordinates": True,
            "vehicle_config": dict(
                show_navi_mark=False,
                no_wheel_friction=True,
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50)
            ),
            "data_directory": AssetLoader.file_path("nuplan", return_raw_style=False),
        }
    )
    success = []
    env.reset(force_seed=0)
    while True:
        env.reset(force_seed=env.current_seed + 1)
        for t in range(10000):
            o, r, d, info = env.step([0, 0])
            assert env.observation_space.contains(o)
            c_lane = env.vehicle.lane
            long, lat, = c_lane.local_coordinates(env.vehicle.position)
            # if env.config["use_render"]:
            env.render(
                text={
                    # "obs_shape": len(o),
                    # "lateral": env.observations["default_agent"].lateral_dist,
                    "seed": env.engine.global_seed + env.config["start_scenario_index"],
                    # "reward": r,
                }
                # mode="topdown"
            )

            if d and info["arrive_dest"]:
                print("seed:{}, success".format(env.engine.global_random_seed))
                break
