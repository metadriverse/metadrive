from metadrive.constants import DEFAULT_AGENT
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.base_env import BaseEnv
from metadrive.manager.waymo_data_manager import WaymoDataManager
from metadrive.manager.waymo_map_manager import WaymoMapManager
from metadrive.manager.waymo_traffic_manager import WaymoTrafficManager
from metadrive.policy.idm_policy import WaymoIDMPolicy
from metadrive.utils import get_np_random

try:
    from metadrive.utils.waymo_map_utils import AgentType
    from metadrive.utils.waymo_map_utils import RoadEdgeType
    from metadrive.utils.waymo_map_utils import RoadLineType
finally:
    pass

WAYMO_ENV_CONFIG = dict(
    # ===== Map Config =====
    waymo_data_directory=AssetLoader.file_path("waymo", "processed", return_raw_style=False),
    case_num=1,

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
        super(WaymoEnv, self).setup_engine()
        self.engine.register_manager("map_manager", WaymoMapManager())
        self.engine.register_manager("traffic_manager", WaymoTrafficManager())
        self.engine.register_manager("data_manager", WaymoDataManager())

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
        return False, {}

    def _reset_global_seed(self, force_seed=None):
        current_seed = force_seed if force_seed is not None else get_np_random(None).randint(0, int(
            self.config["case_num"]))
        self.seed(current_seed)


if __name__ == "__main__":
    env = WaymoEnv(
        {
            "use_render": True,
            "manual_control": True,
            "debug_static_world": True,
            "debug": True,
            # "agent_policy": WaymoIDMPolicy,
            "enable_idm_lane_change": False,
            # "pstats":True
        }
    )
    env.reset()
    while True:
        env.step([0, 0])
        c_lane = env.vehicle.lane
        long, lat = c_lane.local_coordinates(env.vehicle.position)
        env.render(
            text={
                "lane_index": env.vehicle.lane_index,
                "current_ckpt_index": env.vehicle.navigation.current_checkpoint_lane_index,
                "next_ckpt_index": env.vehicle.navigation.next_checkpoint_lane_index,
                "ckpts": env.vehicle.navigation.checkpoints,
                "final_lane": env.vehicle.navigation.final_lane.index,
                "lane_heading": c_lane.heading_theta_at(long),
                "long": long,
                "lat": lat,
                "v_heading": env.vehicle.heading_theta
            }
        )
