import logging

from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.base_env import BaseEnv
from metadrive.manager.waymo_data_manager import WaymoDataManager
from metadrive.manager.waymo_map_manager import WaymoMapManager
from metadrive.manager.waymo_traffic_manager import WaymoTrafficManager
from metadrive.policy.idm_policy import WaymoIDMPolicy
from metadrive.utils import get_np_random

try:
    from metadrive.utils.waymo_utils.waymo_utils import AgentType
    from metadrive.utils.waymo_utils.waymo_utils import RoadEdgeType
    from metadrive.utils.waymo_utils.waymo_utils import RoadLineType
finally:
    pass

WAYMO_ENV_CONFIG = dict(
    # ===== Map Config =====
    waymo_data_directory=AssetLoader.file_path("waymo", "processed", return_raw_style=False),
    case_num=60,
    store_map=True,
    no_traffic=True,
    # ===== Traffic =====
    # traffic_density=0.1,
    # need_inverse_traffic=False,
    # traffic_mode=TrafficMode.Trigger,  # "Respawn", "Trigger"
    # random_traffic=False,  # Traffic is randomized at default.
    # # this will update the vehicle_config and set to traffic
    # traffic_vehicle_config=dict(
    #     show_navi_mark=False,
    #     show_dest_mark=False,
    #     enable_reverse=False,
    #     show_lidar=False,
    #     show_lane_line_detector=False,
    #     show_side_detector=False,
    # ),

    # ===== Agent config =====
    target_vehicle_configs={DEFAULT_AGENT: dict(spawn_lane_index=313, destination=265)},
    enable_idm_lane_change=False,

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

    def _merge_extra_config(self, config):
        config = self.default_config().update(config, allow_add_new_key=False)
        return config

    def _get_observations(self):
        return {self.DEFAULT_AGENT: self.get_single_observation(self.config["vehicle_config"])}

    def setup_engine(self):
        self.in_stop = False
        super(WaymoEnv, self).setup_engine()
        self.engine.register_manager("data_manager", WaymoDataManager())
        self.engine.register_manager("map_manager", WaymoMapManager())
        if not self.config["no_traffic"]:
            self.engine.register_manager("traffic_manager", WaymoTrafficManager())
        self.engine.accept("s", self.stop)

    def step(self, actions):
        ret = super(WaymoEnv, self).step(actions)
        while self.in_stop:
            self.engine.taskMgr.step()
        return ret

    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: name of this base vehicle
        :return: reward, reward info
        """
        return 0, {}

    def cost_function(self, vehicle_id: str):
        return 0, {}

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

    def _reset_global_seed(self, force_seed=None):
        current_seed = force_seed if force_seed is not None else get_np_random(None).randint(
            0, int(self.config["case_num"])
        )
        self.seed(current_seed)

    def _is_out_of_road(self, vehicle):
        # A specified function to determine whether this vehicle should be done.
        # return vehicle.on_yellow_continuous_line or (not vehicle.on_lane) or vehicle.crash_sidewalk
        ret = vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line or vehicle.crash_sidewalk or not vehicle.on_lane
        return ret

    def stop(self):
        self.in_stop = not self.in_stop


if __name__ == "__main__":
    env = WaymoEnv(
        {
            "use_render": False,
            "agent_policy": WaymoIDMPolicy,
            # "manual_control": True,
            # "debug":True,
            "horizon": 1000,
        }
    )
    success = []
    for i in range(60):
        try:
            env.reset(force_seed=i)
            while True:
                o, r, d, info = env.step([0, 0])
                c_lane = env.vehicle.lane
                long, lat, = c_lane.local_coordinates(env.vehicle.position)
                if env.config["use_render"]:
                    env.render(
                        text={
                            "routing_lane_idx": env.engine._object_policies[env.vehicle.id].routing_target_lane.index,
                            "lane_index": env.vehicle.lane_index,
                            "current_ckpt_index": env.vehicle.navigation.current_checkpoint_lane_index,
                            "next_ckpt_index": env.vehicle.navigation.next_checkpoint_lane_index,
                            "ckpts": env.vehicle.navigation.checkpoints,
                            "lane_heading": c_lane.heading_theta_at(long),
                            "long": long,
                            "lat": lat,
                            "v_heading": env.vehicle.heading_theta
                        }
                    )

                if d or env.episode_steps > 1000:
                    if info["arrive_dest"] and env.episode_steps > 100:
                        success.append(i)
                        print("Success, Seed: {}".format(i))
                    else:
                        print("IDM Fail, Seed: {}".format(i))
                        print(info)
                    break
        except:
            print("No Route, Fail, Seed: {}".format(i))
    print(success)
