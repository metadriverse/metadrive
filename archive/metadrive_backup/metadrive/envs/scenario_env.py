"""
This environment can load all scenarios exported from other environments via env.export_scenarios()
"""

import numpy as np

from metadrive.component.navigation_module.trajectory_navigation import TrajectoryNavigation
from metadrive.constants import TerminationState
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.base_env import BaseEnv
from metadrive.manager.scenario_agent_manager import ScenarioAgentManager
from metadrive.manager.scenario_curriculum_manager import ScenarioCurriculumManager
from metadrive.manager.scenario_data_manager import ScenarioDataManager, ScenarioOnlineDataManager
from metadrive.manager.scenario_light_manager import ScenarioLightManager
from metadrive.manager.scenario_map_manager import ScenarioMapManager
from metadrive.manager.scenario_traffic_manager import ScenarioTrafficManager
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.policy.waypoint_policy import WaypointPolicy
from metadrive.utils import get_np_random
from metadrive.utils.math import wrap_to_pi

SCENARIO_ENV_CONFIG = dict(
    # ===== Scenario Config =====
    data_directory=AssetLoader.file_path("nuscenes", unix_style=False),
    start_scenario_index=0,

    # Set num_scenarios=-1 to load all scenarios in the data directory.
    num_scenarios=3,
    sequential_seed=False,  # Whether to set seed (the index of map) sequentially across episodes
    worker_index=0,  # Allowing multi-worker sampling with Rllib
    num_workers=1,  # Allowing multi-worker sampling with Rllib

    # ===== Curriculum Config =====
    curriculum_level=1,  # i.e. set to 5 to split the data into 5 difficulty level
    episodes_to_evaluate_curriculum=None,
    target_success_rate=0.8,

    # ===== Map Config =====
    store_map=True,
    store_data=True,
    need_lane_localization=True,
    no_map=False,
    map_region_size=1024,
    cull_lanes_outside_map=True,

    # ===== Scenario =====
    no_traffic=False,  # nothing will be generated including objects/pedestrian/vehicles
    no_static_vehicles=False,  # static vehicle will be removed
    no_light=False,  # no traffic light
    reactive_traffic=False,  # turn on to enable idm traffic
    filter_overlapping_car=True,  # If in one frame a traffic vehicle collides with ego car, it won't be created.
    default_vehicle_in_traffic=False,
    skip_missing_light=True,
    static_traffic_object=True,
    show_sidewalk=False,
    even_sample_vehicle_class=None,  # Deprecated.

    # ===== Agent config =====
    vehicle_config=dict(
        navigation_module=TrajectoryNavigation,
        lidar=dict(num_lasers=120, distance=50),
        lane_line_detector=dict(num_lasers=0, distance=50),
        side_detector=dict(num_lasers=12, distance=50),
    ),
    # If set_static=True, then the agent will not "fall from the sky". This will be helpful if you want to
    # capture per-frame data for the agent (for example for collecting static sensor data).
    # However, the physics simulation of the agent will be disable too. So in the visualization, the image will be
    # very chunky as the agent will suddenly move to the next position for each step.
    # Set to False for better visualization.
    set_static=False,

    # ===== Reward Scheme =====
    # See: https://github.com/metadriverse/metadrive/issues/283
    success_reward=5.0,
    out_of_road_penalty=5.0,
    on_lane_line_penalty=1.,
    crash_vehicle_penalty=1.,
    crash_object_penalty=1.0,
    crash_human_penalty=1.0,
    driving_reward=1.0,
    steering_range_penalty=0.5,
    heading_penalty=1.0,
    lateral_penalty=.5,
    max_lateral_dist=4,
    no_negative_reward=True,

    # ===== Cost Scheme =====
    crash_vehicle_cost=1.0,
    crash_object_cost=1.0,
    out_of_road_cost=1.0,
    crash_human_cost=1.0,

    # ===== Termination Scheme =====
    out_of_route_done=False,
    crash_vehicle_done=False,
    crash_object_done=False,
    crash_human_done=False,
    relax_out_of_road_done=True,

    # ===== others =====
    allowed_more_steps=None,  # horizon, None=infinite
    top_down_show_real_size=False,
    use_bounding_box=False,  # Set True to use a cube in visualization to represent every dynamic objects.
)

SCENARIO_WAYPOINT_ENV_CONFIG = dict(
    # How many waypoints will be used at each environmental step. Checkout ScenarioWaypointEnv for details.
    waypoint_horizon=5,
    agent_policy=WaypointPolicy,

    # Must set this to True, otherwise the agent will drift away from the waypoint when doing
    # "self.engine.step(self.config["decision_repeat"])" in "_step_simulator".
    set_static=True,
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
        self.start_index = self.config["start_scenario_index"]

    def _post_process_config(self, config):
        config = super(ScenarioEnv, self)._post_process_config(config)
        if config["use_bounding_box"]:
            config["vehicle_config"]["random_color"] = True
            config["vehicle_config"]["vehicle_model"] = "varying_dynamics_bounding_box"
            config["agent_configs"]["default_agent"]["use_special_color"] = True
            config["agent_configs"]["default_agent"]["vehicle_model"] = "varying_dynamics_bounding_box"
        return config

    def _get_agent_manager(self):
        return ScenarioAgentManager(init_observations=self._get_observations())

    def setup_engine(self):
        super(ScenarioEnv, self).setup_engine()
        self.engine.register_manager("data_manager", ScenarioDataManager())
        self.engine.register_manager("map_manager", ScenarioMapManager())
        if not self.config["no_traffic"]:
            self.engine.register_manager("traffic_manager", ScenarioTrafficManager())
        if not self.config["no_light"]:
            self.engine.register_manager("light_manager", ScenarioLightManager())
        self.engine.register_manager("curriculum_manager", ScenarioCurriculumManager())

    def done_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        done = False
        max_step = self.config["horizon"] is not None and self.episode_lengths[vehicle_id] >= self.config["horizon"]
        done_info = {
            TerminationState.CRASH_VEHICLE: vehicle.crash_vehicle,
            TerminationState.CRASH_OBJECT: vehicle.crash_object,
            TerminationState.CRASH_BUILDING: vehicle.crash_building,
            TerminationState.CRASH_HUMAN: vehicle.crash_human,
            TerminationState.CRASH_SIDEWALK: vehicle.crash_sidewalk,
            TerminationState.OUT_OF_ROAD: self._is_out_of_road(vehicle) or vehicle.navigation.route_completion < -0.1,
            TerminationState.SUCCESS: self._is_arrive_destination(vehicle),
            TerminationState.MAX_STEP: max_step,
            TerminationState.ENV_SEED: self.current_seed,
            # TerminationState.CURRENT_BLOCK: self.agent.navigation.current_road.block_ID(),
            # crash_vehicle=False, crash_object=False, crash_building=False, out_of_road=False, arrive_dest=False,
        }

        # for compatibility
        # crash almost equals to crashing with vehicles
        done_info[TerminationState.CRASH] = (
            done_info[TerminationState.CRASH_VEHICLE] or done_info[TerminationState.CRASH_OBJECT]
            or done_info[TerminationState.CRASH_BUILDING] or done_info[TerminationState.CRASH_SIDEWALK]
            or done_info[TerminationState.CRASH_HUMAN]
        )

        def msg(reason):
            return "Episode ended! Scenario Index: {} Scenario id: {} Reason: {}.".format(
                self.current_seed, self.engine.data_manager.current_scenario_id, reason
            )

        if done_info[TerminationState.SUCCESS]:
            done = True
            self.logger.debug(msg("arrive_dest"), extra={"log_once": True})
        elif done_info[TerminationState.OUT_OF_ROAD]:
            done = True
            self.logger.debug(msg("out_of_road"), extra={"log_once": True})
        elif done_info[TerminationState.CRASH_HUMAN] and self.config["crash_human_done"]:
            done = True
            self.logger.debug(msg("crash human"), extra={"log_once": True})
        elif done_info[TerminationState.CRASH_VEHICLE] and self.config["crash_vehicle_done"]:
            done = True
            self.logger.debug(msg("crash vehicle"), extra={"log_once": True})
        elif done_info[TerminationState.CRASH_OBJECT] and self.config["crash_object_done"]:
            done = True
            self.logger.debug(msg("crash object"), extra={"log_once": True})
        elif done_info[TerminationState.CRASH_BUILDING] and self.config["crash_object_done"]:
            done = True
            self.logger.debug(msg("crash building"), extra={"log_once": True})
        elif done_info[TerminationState.MAX_STEP]:
            if self.config["truncate_as_terminate"]:
                done = True
            self.logger.debug(msg("max step"), extra={"log_once": True})
        elif self.config["allowed_more_steps"] and self.episode_lengths[vehicle_id] >= \
            self.engine.data_manager.current_scenario_length + self.config["allowed_more_steps"]:
            if self.config["truncate_as_terminate"]:
                done = True
            done_info[TerminationState.MAX_STEP] = True
            self.logger.debug(msg("more step than original episode"), extra={"log_once": True})

        # log data to curriculum manager
        self.engine.curriculum_manager.log_episode(
            done_info[TerminationState.SUCCESS], vehicle.navigation.route_completion
        )

        return done, done_info

    def cost_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict(num_crash_object=0, num_crash_human=0, num_crash_vehicle=0, num_on_line=0)
        step_info["cost"] = 0
        if vehicle.on_yellow_continuous_line or vehicle.crash_sidewalk or vehicle.on_white_continuous_line:
            # step_info["cost"] += self.config["out_of_road_cost"]
            step_info["num_on_line"] = 1
        if self._is_out_of_road(vehicle):
            step_info["cost"] += self.config["out_of_road_cost"]
        if vehicle.crash_vehicle:
            step_info["cost"] += self.config["crash_vehicle_cost"]
            step_info["crash_vehicle_cost"] = self.config["crash_vehicle_cost"]
            step_info["num_crash_vehicle"] = 1
        if vehicle.crash_object:
            step_info["cost"] += self.config["crash_object_cost"]
            step_info["num_crash_object"] = 1
        if vehicle.crash_human:
            step_info["cost"] += self.config["crash_human_cost"]
            step_info["num_crash_human"] = 1
        return step_info['cost'], step_info

    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        vehicle = self.agents[vehicle_id]
        step_info = dict()

        # Reward for moving forward in current lane
        current_lane = vehicle.lane
        long_last = vehicle.navigation.last_longitude
        long_now = vehicle.navigation.current_longitude
        lateral_now = vehicle.navigation.current_lateral

        # dense driving reward
        reward = 0
        reward += self.config["driving_reward"] * (long_now - long_last)

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        lateral_factor = abs(lateral_now) / self.config["max_lateral_dist"]
        lateral_penalty = -lateral_factor * self.config["lateral_penalty"]
        reward += lateral_penalty

        # heading diff
        ref_line_heading = vehicle.navigation.current_heading_theta_at_long
        heading_diff = wrap_to_pi(abs(vehicle.heading_theta - ref_line_heading)) / np.pi
        heading_penalty = -heading_diff * self.config["heading_penalty"]
        reward += heading_penalty

        # steering_range
        steering = abs(vehicle.current_action[0])
        allowed_steering = (1 / max(vehicle.speed, 1e-2))
        overflowed_steering = min((allowed_steering - steering), 0)
        steering_range_penalty = overflowed_steering * self.config["steering_range_penalty"]
        reward += steering_range_penalty

        if self.config["no_negative_reward"]:
            reward = max(reward, 0)

        # crash penalty
        if vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        if vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        if vehicle.crash_human:
            reward = -self.config["crash_human_penalty"]
        # lane line penalty
        if vehicle.on_yellow_continuous_line or vehicle.crash_sidewalk or vehicle.on_white_continuous_line:
            reward = -self.config["on_lane_line_penalty"]

        step_info["step_reward"] = reward

        # termination reward
        if self._is_arrive_destination(vehicle):
            reward = self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]

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
        step_info["lateral_dist"] = lateral_now

        step_info["step_reward_lateral"] = lateral_penalty
        step_info["step_reward_heading"] = heading_penalty
        step_info["step_reward_action_smooth"] = steering_range_penalty
        return reward, step_info

    # Compute state difference metrics for reward
    # TODO LQY: Shall we use state difference as reward?
    # data = self.engine.data_manager.current_scenario
    # agent_xy = vehicle.position
    # if vehicle_id == "sdc" or vehicle_id == "default_agent":
    #     native_vid = data[ScenarioDescription.METADATA][ScenarioDescription.SDC_ID]
    # else:
    #     native_vid = vehicle_id
    #
    # if native_vid in data["tracks"] and len(data["tracks"][native_vid]) > 0:
    #     expert_state_list = data["tracks"][native_vid]["state"]
    #
    #     mask = expert_state_list["valid"]
    #     largest_valid_index = np.max(np.where(mask == True)[0])
    #
    #     if self.episode_step > largest_valid_index:
    #         current_step = largest_valid_index
    #     else:
    #         current_step = self.episode_step
    #
    #     while mask[current_step] == 0.0:
    #         current_step -= 1
    #         if current_step == 0:
    #             break
    #
    #     expert_xy = expert_state_list["position"][current_step][:2]
    #     diff = agent_xy - expert_xy
    #     dist = norm(diff[0], diff[1])
    #     step_info["distance_error"] = dist
    #
    #     last_state = expert_state_list["position"][largest_valid_index]
    #     last_expert_xy = last_state[:2]
    #     diff = agent_xy - last_expert_xy
    #     last_dist = norm(diff[0], diff[1])
    #     step_info["distance_error_final"] = last_dist

    # reward = reward - self.config["distance_penalty"] * dist

    # if hasattr(vehicle, "_dynamics_mode"):
    #     step_info["dynamics_mode"] = vehicle._dynamics_mode

    @staticmethod
    def _is_arrive_destination(vehicle):
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
            lat = abs(vehicle.navigation.current_lateral)
            done = lat > self.config["max_lateral_dist"]
            done = done
            return done

        done = vehicle.crash_sidewalk or vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line
        if self.config["out_of_route_done"]:
            done = done or abs(vehicle.navigation.current_lateral) > 10
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

        assert self.config["start_scenario_index"] <= current_seed < self.config["start_scenario_index"] + \
               self.config["num_scenarios"], "Scenario Index (force seed) {} is out of range [{}, {}).".format(
            current_seed, self.config["start_scenario_index"],
            self.config["start_scenario_index"] + self.config["num_scenarios"])
        self.seed(current_seed)


class ScenarioOnlineEnv(ScenarioEnv):
    """
    This environment allow the user to pass in scenario data directly.
    """
    def default_config(cls):
        config = super(ScenarioOnlineEnv, cls).default_config()
        config.update({
            "store_map": False,
        })
        return config

    def __init__(self, config=None):
        super(ScenarioOnlineEnv, self).__init__(config)
        self.lazy_init()

        assert self.config["store_map"] is False, \
            "ScenarioOnlineEnv should not store map. Please set store_map=False in config"

    def setup_engine(self):
        """Overwrite the data_manager by ScenarioOnlineDataManager"""
        super().setup_engine()
        self.engine.update_manager("data_manager", ScenarioOnlineDataManager())

    def set_scenario(self, scenario_data):
        """Please call this function before env.reset()"""
        self.engine.data_manager.set_scenario(scenario_data)


class ScenarioWaypointEnv(ScenarioEnv):
    """
    This environment use WaypointPolicy. Even though the environment still runs in 10 HZ, we allow the external
    waypoint generator generates up to 5 waypoints at each step (controlled by config "waypoint_horizon").
    Say at step t, we receive 5 waypoints. Then we will set the agent states for t+1, t+2, t+3, t+4, t+5 if at
    t+1 ~ t+4 no additional waypoints are received. Here is the full timeline:

    step t=0: env.reset(), initial positions/obs are sent out. This corresponds to the t=0 or t=10 in WOMD dataset
    (TODO: we should allow control on the meaning of the t=0)
    step t=1: env.step(), agent receives 5 waypoints, we will record the waypoint sequences. Set agent state for t=1,
        and send out the obs for t=1.
    step t=2: env.step(), it's possible to get action=None, which means the agent will use the cached waypoint t=2,
        and set the agent state for t=2. The obs for t=2 will be sent out. If new waypoints are received, we will \
        instead set agent state to the first new waypoint.
    step t=3: ... continues the loop and receives action=None or new waypoints.
    step t=4: ...
    step t=5: ...
    step t=6: if we only receive action at t=1, and t=2~t=5 are all None, then this step will force to receive
        new waypoints. We will set the agent state to the first new waypoint.

    Most of the functions are implemented in WaypointPolicy.
    """
    @classmethod
    def default_config(cls):
        config = super(ScenarioWaypointEnv, cls).default_config()
        config.update(SCENARIO_WAYPOINT_ENV_CONFIG)
        return config

    def _post_process_config(self, config):
        ret = super(ScenarioWaypointEnv, self)._post_process_config(config)
        assert config["set_static"], "Waypoint policy requires set_static=True"
        return ret


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
            "num_scenarios": 3,
            "set_static": True,
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
            "data_directory": AssetLoader.file_path("nuscenes", unix_style=False),
        }
    )
    success = []
    env.reset(seed=0)
    while True:
        env.reset(seed=env.current_seed + 1)
        for t in range(10000):
            o, r, tm, tc, info = env.step([0, 0])
            assert env.observation_space.contains(o)
            c_lane = env.agent.lane
            long, lat, = c_lane.local_coordinates(env.agent.position)
            # if env.config["use_render"]:
            env.render(
                text={
                    # "obs_shape": len(o),
                    "seed": env.engine.global_seed + env.config["start_scenario_index"],
                    # "reward": r,
                }
                # mode="topdown"
            )

            if (tm or tc) and info["arrive_dest"]:
                print("seed:{}, success".format(env.engine.global_random_seed))
                break
