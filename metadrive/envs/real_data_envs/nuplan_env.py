import logging

import numpy as np

from metadrive.component.vehicle_navigation_module.trajectory_navigation import NuPlanTrajectoryNavigation
from metadrive.constants import TerminationState
from metadrive.envs.base_env import BaseEnv
from metadrive.manager.nuplan_data_manager import NuPlanDataManager
from metadrive.manager.nuplan_map_manager import NuPlanMapManager
from metadrive.manager.nuplan_traffic_manager import NuPlanTrafficManager
from metadrive.obs.real_env_observation import NuPlanObservation
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.policy.replay_policy import NuPlanReplayEgoCarPolicy
from metadrive.utils import clip
from metadrive.utils import get_np_random

NUPLAN_ENV_CONFIG = dict(
    # ===== Dataset Config =====
    # These parameters remain the same as the meaning of those in nuplan-devkit
    DATASET_PARAMS=[
        'scenario_builder=nuplan_mini',  # use nuplan mini database (2.5h of 8 autolabeled logs in Las Vegas)
        'scenario_filter=one_continuous_log',  # simulate only one log
        "scenario_filter.log_names=['2021.07.16.20.45.29_veh-35_01095_01486']",
        'scenario_filter.limit_total_scenarios=2800',  # use 2 total scenarios
    ],
    start_case_index=0,
    case_num=100,
    store_map=True,
    store_map_buffer_size=200,
    sequential_seed=False,

    # ===== Traffic =====
    no_traffic=False,
    replay=True,
    no_static_traffic_vehicle=False,

    # ===== Agent config =====
    vehicle_config=dict(
        lidar=dict(num_lasers=120, distance=50),
        lane_line_detector=dict(num_lasers=12, distance=50),
        side_detector=dict(num_lasers=120, distance=50),
        show_dest_mark=True,
        navigation_module=NuPlanTrajectoryNavigation,
    ),
    use_nuplan_observation=True,

    # ===== Reward Scheme =====
    # See: https://github.com/metadriverse/metadrive/issues/283
    success_reward=10.0,
    out_of_road_penalty=10.0,
    crash_vehicle_penalty=10.0,
    crash_object_penalty=1.0,
    driving_reward=1.0,
    speed_reward=0.1,
    use_lateral_reward=False,
    horizon=500,

    # ===== Cost Scheme =====
    crash_vehicle_cost=1.0,
    crash_object_cost=1.0,
    out_of_road_cost=1.0,

    # ===== Termination Scheme =====
    out_of_route_done=False,
    crash_vehicle_done=True,
)


class NuPlanEnv(BaseEnv):
    @classmethod
    def default_config(cls):
        config = super(NuPlanEnv, cls).default_config()
        config.update(NUPLAN_ENV_CONFIG)
        return config

    def __init__(self, config=None):
        super(NuPlanEnv, self).__init__(config)

    def _merge_extra_config(self, config):
        # config = self.default_config().update(config, allow_add_new_key=True)
        config = self.default_config().update(config, allow_add_new_key=False)
        return config

    def _get_observations(self):
        return {self.DEFAULT_AGENT: self.get_single_observation(self.config["vehicle_config"])}

    def get_single_observation(self, vehicle_config):
        if self.config["use_nuplan_observation"]:
            o = NuPlanObservation(vehicle_config)
        else:
            o = LidarStateObservation(vehicle_config)
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
        super(NuPlanEnv, self).setup_engine()
        self.engine.register_manager("data_manager", NuPlanDataManager())
        self.engine.register_manager("map_manager", NuPlanMapManager())
        if not self.config["no_traffic"]:
            if not self.config['replay']:
                raise ValueError
                self.engine.register_manager("traffic_manager", NuPlanIDMTrafficManager())
            else:
                self.engine.register_manager("traffic_manager", NuPlanTrafficManager())
        self.engine.accept("p", self.stop)
        self.engine.accept("q", self.switch_to_third_person_view)
        self.engine.accept("b", self.switch_to_top_down_view)
        self.engine.accept("]", self.next_seed_reset)
        self.engine.accept("[", self.last_seed_reset)

    def next_seed_reset(self):
        self.reset(self.current_seed + 1)

    def last_seed_reset(self):
        self.reset(self.current_seed - 1)

    def step(self, actions):
        ret = super(NuPlanEnv, self).step(actions)
        while self.in_stop:
            self.engine.taskMgr.step()
        return ret

    def done_function(self, vehicle_id: str):
        vehicle = self.vehicles[vehicle_id]
        done = False
        done_info = dict(
            crash_vehicle=False, crash_object=False, crash_building=False, out_of_road=False, arrive_dest=False
        )
        # if np.linalg.norm(vehicle.position - self.engine.map_manager.sdc_dest_point) < 5 \
        #         or vehicle.lane.index in self.engine.map_manager.sdc_destinations:
        if self._is_arrive_destination(vehicle):
            done = True
            logging.info("Episode ended! Reason: arrive_dest.")
            done_info[TerminationState.SUCCESS] = True
        if self._is_out_of_road(vehicle):
            done = True
            logging.info("Episode ended! Reason: out_of_road.")
            done_info[TerminationState.OUT_OF_ROAD] = True
        if vehicle.crash_vehicle and self.config["crash_vehicle_done"]:
            done = True
            logging.info("Episode ended! Reason: crash vehicle ")
            done_info[TerminationState.CRASH_VEHICLE] = True
        if vehicle.crash_object:
            done = True
            done_info[TerminationState.CRASH_OBJECT] = True
            logging.info("Episode ended! Reason: crash object ")
        if vehicle.crash_building:
            done = True
            done_info[TerminationState.CRASH_BUILDING] = True
            logging.info("Episode ended! Reason: crash building ")

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

        reward = 0
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor

        step_info["step_reward"] = reward

        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        return reward, step_info

    def _is_arrive_destination(self, vehicle):
        return True if np.linalg.norm(vehicle.position - self.engine.map_manager.sdc_dest_point) < 5 else False

    def _reset_global_seed(self, force_seed=None):
        if force_seed is not None:
            current_seed = force_seed
        elif self.config["sequential_seed"]:
            current_seed = self.engine.global_seed
            if current_seed is None:
                current_seed = self.config["start_case_index"]
            else:
                current_seed += 1
            if current_seed >= self.config["start_case_index"] + int(self.config["case_num"]):
                current_seed = self.config["start_case_index"]
        else:
            current_seed = get_np_random(None).randint(
                self.config["start_case_index"], self.config["start_case_index"] + int(self.config["case_num"])
            )

        assert self.config["start_case_index"] <= current_seed < \
               self.config["start_case_index"] + self.config["case_num"], "Force seed range Error!"
        self.seed(current_seed)

    def _is_out_of_road(self, vehicle):
        # A specified function to determine whether this vehicle should be done.
        done = vehicle.crash_sidewalk or vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line
        if self.config["out_of_route_done"]:
            done = done or self.observations[self.DEFAULT_AGENT].lateral_dist > 10
        return done
        # ret = vehicle.crash_sidewalk
        # return ret

    def stop(self):
        self.in_stop = not self.in_stop


if __name__ == "__main__":
    env = NuPlanEnv(
        {
            "use_render": True,
            "agent_policy": NuPlanReplayEgoCarPolicy,
            "manual_control": True,
            "replay": True,
            "no_traffic": False,
            # "debug": True,
            # "debug_static_world": True,
            # "no_traffic":True,
            # "start_case_index": 192,
            # "start_case_index": 1000,
            # "waymo_data_directory": "E:\\PAMI_waymo_data\\idm_filtered\\test",
            "window_size": (2400, 1600),
            "start_case_index": 200,
            "case_num": 2000,
            "horizon": 1000,
            "vehicle_config": dict(
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50),
                # need_navigation=False
            ),
            # "show_interface":False
        }
    )
    success = []
    for seed in range(300, 2000):
        env.reset(force_seed=302)
        for i in range(env.engine.data_manager.current_scenario_length * 10):
            o, r, d, info = env.step([0, 0])
            # assert env.observation_space.contains(o)
            # c_lane = env.vehicle.lane
            # long, lat, = c_lane.local_coordinates(env.vehicle.position)
            # if env.config["use_render"]:
            #     env.render(
            #         text={
            #             # "routing_lane_idx": env.engine._object_policies[env.vehicle.id].routing_target_lane.index,
            #             # "lane_index": env.vehicle.lane_index,
            #             # "current_ckpt_index": env.vehicle.navigation.current_checkpoint_lane_index,
            #             # "next_ckpt_index": env.vehicle.navigation.next_checkpoint_lane_index,
            #             # "ckpts": env.vehicle.navigation.checkpoints,
            #             # "lane_heading": c_lane.heading_theta_at(long),
            #             # "long": long,
            #             # "lat": lat,
            #             # "v_heading": env.vehicle.heading_theta,
            #             "obs_shape": len(o),
            #             "lateral": env.observations["default_agent"].lateral_dist,
            #             "seed": env.engine.global_seed + env.config["start_case_index"],
            #             "reward": r,
            #         }
            #     )
            #
            # if d:
            #     if info["arrive_dest"]:
            #         print("seed:{}, success".format(env.engine.global_random_seed))
            #     break
