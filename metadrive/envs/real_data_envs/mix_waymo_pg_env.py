from typing import Union
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.engine_utils import engine_initialized
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.manager.map_manager import MapManager
from metadrive.manager.traffic_manager import TrafficManager
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.manager.waymo_map_manager import WaymoMapManager
from metadrive.manager.waymo_traffic_manager import WaymoTrafficManager
from metadrive.manager.waymo_data_manager import WaymoDataManager
from metadrive.utils import Config, merge_dicts, get_np_random, concat_step_infos

MIX_WAYMO_PG_ENV_CONFIG = dict(
    # ===== Waymo Map Config =====
    waymo_data_directory=AssetLoader.file_path("waymo", "processed", return_raw_style=False),
    start_case_index=0,
    case_num=50,
    store_map=True,

    # ===== Waymo Traffic Config =====
    no_traffic=False,
    case_start_index=0,
    case_end_index=-1,
    replay=True,

    # ===== PG Map config =====
    start_seed=0,
    environment_num=50,

    # ===== PG Map Config =====
    block_num=1,  # block_num
    random_lane_width=False,
    random_lane_num=False,

    # ===== PG Traffic =====
    traffic_density=0.1,
    need_inverse_traffic=False,
    traffic_mode=TrafficMode.Trigger,  # "Respawn", "Trigger"
    random_traffic=False,  # Traffic is randomized at default.
    # this will update the vehicle_config and set to traffic
    traffic_vehicle_config=dict(
        show_navi_mark=False,
        show_dest_mark=False,
        enable_reverse=False,
        show_lidar=False,
        show_lane_line_detector=False,
        show_side_detector=False,
    ),

    # ===== engine config =====
    force_destroy=True,

)


class MixWaymoPGEnv(WaymoEnv):
    @classmethod
    def default_config(cls):
        config = super(MixWaymoPGEnv, cls).default_config()
        MIX_WAYMO_PG_ENV_CONFIG.update(dict(
            map_config={
                BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
                BaseMap.GENERATE_CONFIG: MIX_WAYMO_PG_ENV_CONFIG["block_num"],  # it can be a file path / block num / block ID sequence
                BaseMap.LANE_WIDTH: 3.5,
                BaseMap.LANE_NUM: 3,
                "exit_length": 50,
            }))
        config.update(MIX_WAYMO_PG_ENV_CONFIG)
        return config

    def __init__(self, config):
        super(MixWaymoPGEnv, self).__init__(config)
        self.waymo_map_manager = None
        self.waymo_traffic_manager = None
        self.pg_map_manager = None
        self.pg_traffic_manager = None

        self.total_environment = self.config["case_num"] + self.config["environment_num"]
        self.real_data_ratio = self.config["case_num"] / self.total_environment
        self.is_current_real_data = True

    def setup_engine(self):
        # Initialize all managers
        self.waymo_map_manager = WaymoMapManager()
        self.waymo_traffic_manager = WaymoTrafficManager()

        self.pg_map_manager = MapManager()
        self.pg_traffic_manager = TrafficManager()

        self.in_stop = False
        super(WaymoEnv, self).setup_engine()
        self.engine.register_manager("data_manager", WaymoDataManager())
        self.engine.register_manager("map_manager", self.waymo_map_manager)
        if not self.config["no_traffic"]:
            self.engine.register_manager("traffic_manager", self.waymo_traffic_manager)
        self.engine.accept("s", self.stop)
        self.engine.accept("q", self.switch_to_third_person_view)
        self.engine.accept("b", self.switch_to_top_down_view)

    def reset(self, force_seed: Union[None, int] = None):
        if engine_initialized():
            # must lazy initialize at first
            if get_np_random(None).rand() < self.real_data_ratio:
                # change to real environment
                self.engine.update_manager("map_manager",
                                           self.waymo_map_manager,
                                           destroy_previous_manager=False)
                self.engine.update_manager("traffic_manager",
                                           self.waymo_traffic_manager,
                                           destroy_previous_manager=False)
                self.is_current_real_data = True
            else:
                self.is_current_real_data = False
                # change to PG environment
                self.engine.update_manager("map_manager",
                                           self.pg_map_manager,
                                           destroy_previous_manager=False)
                self.engine.update_manager("traffic_manager",
                                           self.pg_traffic_manager,
                                           destroy_previous_manager=False)
                self.config["target_vehicle_configs"]["default_agent"]["spawn_lane_index"] = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0)
                self.config["target_vehicle_configs"]["default_agent"]["destination"] = None
        return super(MixWaymoPGEnv, self).reset()

    def _reset_global_seed(self, force_seed=None):
        current_seed = force_seed if force_seed is not None else get_np_random(None).randint(
            0, self.config["case_num"] if self.is_current_real_data else self.config["environment_num"])
        self.seed(current_seed)


    def done_function(self, vehicle_id: str):
        if self.is_current_real_data:
            return super(MixWaymoPGEnv, self).done_function(vehicle_id)
        else:
            return MetaDriveEnv.done_function(self, vehicle_id)

    def reward_function(self, vehicle_id: str):
        if self.is_current_real_data:
            return super(MixWaymoPGEnv, self).reward_function(vehicle_id)
        else:
            return MetaDriveEnv.reward_function(self, vehicle_id)

    def _is_out_of_road(self, vehicle):
        if self.is_current_real_data:
            return super(MixWaymoPGEnv, self)._is_out_of_road(vehicle)
        else:
            return MetaDriveEnv._is_out_of_road(self, vehicle)



if __name__ == "__main__":
    env = MixWaymoPGEnv(dict(manual_control=True,
                             use_render=True,
                             waymo_data_directory="E:\\PAMI_waymo_data\\idm_filtered\\validation",
                             case_num=1,
                             start_case=32,
                             environment_num=1))
    env.reset()
    while True:
        o, r, d, i = env.step(env.action_space.sample())
        if d:
            env.reset()
