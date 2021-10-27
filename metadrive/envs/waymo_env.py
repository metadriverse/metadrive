from metadrive.constants import DEFAULT_AGENT
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.base_env import BaseEnv
from metadrive.manager.waymo_map_manager import WaymoMapManager

try:
    from metadrive.utils.waymo_map_utils import AgentType
    from metadrive.utils.waymo_map_utils import RoadEdgeType
    from metadrive.utils.waymo_map_utils import RoadLineType
finally:
    pass

WAYMO_ENV_CONFIG = dict(
    # TODO add map config and Traffic config
    # ===== Map Config =====
    map_directory=AssetLoader.file_path("waymo", "processed", return_raw_style=False),
    map_num=1,

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
    target_vehicle_configs={DEFAULT_AGENT: dict(spawn_lane_index=118, destination=191)},

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


if __name__ == "__main__":
    env = WaymoEnv({"use_render": True, "manual_control": True, "debug_static_world":True, "debug":True})
    env.reset()
    while True:
        env.step([0, 0])
        env.render(text={
            "lane_index": env.vehicle.lane_index,
            "current_ckpt_index":env.vehicle.navigation.current_checkpoint_lane_index,
            "next_ckpt_index":env.vehicle.navigation.next_checkpoint_lane_index
        })
