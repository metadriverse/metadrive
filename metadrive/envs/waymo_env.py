import logging

from metadrive.constants import TerminationState
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.base_env import BaseEnv
from metadrive.manager.waymo_data_manager import WaymoDataManager
from metadrive.manager.waymo_map_manager import WaymoMapManager
from metadrive.manager.waymo_traffic_manager import WaymoTrafficManager
from metadrive.policy.idm_policy import WaymoIDMPolicy
from metadrive.utils import clip
from metadrive.utils import get_np_random

WAYMO_ENV_CONFIG = dict(
    # ===== Map Config =====
    waymo_data_directory=AssetLoader.file_path("waymo", "processed", return_raw_style=False),
    start_case_index=0,
    case_num=100,
    store_map=True,

    # ===== Traffic =====
    no_traffic=False,
    case_start_index=0,
    case_end_index=-1,
    replay=True,

    # ===== Agent config =====
    vehicle_config=dict(
        lidar=dict(num_lasers=120, distance=50),
        lane_line_detector=dict(num_lasers=12, distance=20),
        side_detector=dict(num_lasers=12, distance=50)
    ),

    # ===== Reward Scheme =====
    # See: https://github.com/decisionforce/metadrive/issues/283
    success_reward=10.0,
    out_of_road_penalty=5.0,
    crash_vehicle_penalty=5.0,
    crash_object_penalty=5.0,
    driving_reward=1.0,
    speed_reward=0.1,
    use_lateral=False,

    # ===== Cost Scheme =====
    crash_vehicle_cost=1.0,
    crash_object_cost=1.0,
    out_of_road_cost=1.0,

    # ===== Termination Scheme =====
    out_of_route_done=False,
)


class WaymoEnv(BaseEnv):
    @classmethod
    def default_config(cls):
        config = super(WaymoEnv, cls).default_config()
        config.update(WAYMO_ENV_CONFIG)
        return config

    def __init__(self, config):
        super(WaymoEnv, self).__init__(config)
        if not self.config["no_traffic"]:
            assert self.config["agent_policy"] is not WaymoIDMPolicy, "WaymoIDM will fail when interacting with traffic"

    def _merge_extra_config(self, config):
        config = self.default_config().update(config, allow_add_new_key=True)
        return config

    def _get_observations(self):
        return {self.DEFAULT_AGENT: self.get_single_observation(self.config["vehicle_config"])}

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
        super(WaymoEnv, self).setup_engine()
        self.engine.register_manager("data_manager", WaymoDataManager())
        self.engine.register_manager("map_manager", WaymoMapManager())
        if not self.config["no_traffic"]:
            self.engine.register_manager("traffic_manager", WaymoTrafficManager())
        self.engine.accept("s", self.stop)
        self.engine.accept("q", self.switch_to_third_person_view)
        self.engine.accept("b", self.switch_to_top_down_view)

    def step(self, actions):
        ret = super(WaymoEnv, self).step(actions)
        while self.in_stop:
            self.engine.taskMgr.step()
        return ret

    def done_function(self, vehicle_id: str):
        vehicle = self.vehicles[vehicle_id]
        done = False
        done_info = dict(
            crash_vehicle=False, crash_object=False, crash_building=False, out_of_road=False, arrive_dest=False
        )
        if vehicle.lane.index in self.engine.map_manager.sdc_destinations:
            done = True
            logging.info("Episode ended! Reason: arrive_dest.")
            done_info[TerminationState.SUCCESS] = True
        if self._is_out_of_road(vehicle):
            done = True
            logging.info("Episode ended! Reason: out_of_road.")
            done_info[TerminationState.OUT_OF_ROAD] = True
        if vehicle.crash_vehicle:
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

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral"]:
            lateral_factor = clip(1 - 2 * abs(lateral_now) / 6, 0.0, 1.0)
        else:
            lateral_factor = 1.0

        reward = 0
        if not vehicle.on_white_continuous_line:
            reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor
            reward += self.config["speed_reward"] * (vehicle.speed / vehicle.max_speed)

        step_info["step_reward"] = reward

        if vehicle.arrive_destination:
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        return reward, step_info

    def _reset_global_seed(self, force_seed=None):
        current_seed = force_seed if force_seed is not None else get_np_random(None).randint(
            0, int(self.config["case_num"])
        )
        self.seed(current_seed)

    def _is_out_of_road(self, vehicle):
        # A specified function to determine whether this vehicle should be done.
        return vehicle.on_yellow_continuous_line or vehicle.crash_sidewalk
        # ret = vehicle.crash_sidewalk
        # return ret

    def stop(self):
        self.in_stop = not self.in_stop


if __name__ == "__main__":
    env = WaymoEnv(
        {
            "use_render": True,
            # "agent_policy": WaymoIDMPolicy,
            "manual_control": True,
            # "debug":True,
            # "no_traffic":True,
            # "start_case_index": 192,
            "case_num": 100,
            "waymo_data_directory": "E:\\hk\\idm_filtered\\validation",
            "horizon": 1000,
            # "vehicle_config": dict(show_lidar=True,
            #                        show_lane_line_detector=True,
            #                        show_side_detector=True)
        }
    )
    success = []
    for i in range(env.config["case_num"]):
        env.reset(force_seed=i)
        while True:
            o, r, d, info = env.step([0, 0])
            assert env.observation_space.contains(o)
            c_lane = env.vehicle.lane
            long, lat, = c_lane.local_coordinates(env.vehicle.position)
            if env.config["use_render"]:
                env.render(
                    text={
                        # "routing_lane_idx": env.engine._object_policies[env.vehicle.id].routing_target_lane.index,
                        # "lane_index": env.vehicle.lane_index,
                        # "current_ckpt_index": env.vehicle.navigation.current_checkpoint_lane_index,
                        # "next_ckpt_index": env.vehicle.navigation.next_checkpoint_lane_index,
                        # "ckpts": env.vehicle.navigation.checkpoints,
                        # "lane_heading": c_lane.heading_theta_at(long),
                        # "long": long,
                        # "lat": lat,
                        # "v_heading": env.vehicle.heading_theta,
                        "seed": env.engine.global_seed + env.config["start_case_index"],
                        "reward": r,
                    }
                )

            if d:
                if info["arrive_dest"]:
                    print("seed:{}, success".format(env.engine.global_random_seed))
                break
