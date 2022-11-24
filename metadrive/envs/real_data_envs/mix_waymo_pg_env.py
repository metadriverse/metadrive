from typing import Union
from collections import defaultdict
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.engine_utils import engine_initialized
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.manager.map_manager import PGMapManager
from metadrive.manager.traffic_manager import PGTrafficManager
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
    traj_start_index=0,
    traj_end_index=-1,
    replay=True,

    # ===== PG Map config =====
    start_seed=0,
    environment_num=50,

    # ===== PG Map Config =====
    block_num=1,  # block_num
    random_lane_width=False,
    random_lane_num=False,

    # ===== PG Traffic =====
    traffic_density=0.2,
    need_inverse_traffic=False,
    traffic_mode=TrafficMode.Hybrid,  # "Respawn", "Trigger"
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
    horizon=2500,
    # use_lateral_reward=True
)


class MixWaymoPGEnv(WaymoEnv):
    @classmethod
    def default_config(cls):
        config = super(MixWaymoPGEnv, cls).default_config()
        MIX_WAYMO_PG_ENV_CONFIG.update(
            dict(
                map_config={
                    BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
                    BaseMap.GENERATE_CONFIG: MIX_WAYMO_PG_ENV_CONFIG["block_num"],
                    # it can be a file path / block num / block ID sequence
                    BaseMap.LANE_WIDTH: 3.5,
                    BaseMap.LANE_NUM: 3,
                    "exit_length": 50,
                }
            )
        )
        config.update(MIX_WAYMO_PG_ENV_CONFIG)
        return config

    def __init__(self, config=None):
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

        self.pg_map_manager = PGMapManager()
        self.pg_traffic_manager = PGTrafficManager()

        self.in_stop = False
        super(WaymoEnv, self).setup_engine()
        if self.real_data_ratio > 0:
            self.is_current_real_data = True
            self.engine.register_manager("data_manager", WaymoDataManager())
            self.engine.register_manager("map_manager", self.waymo_map_manager)
            if not self.config["no_traffic"]:
                self.engine.register_manager("traffic_manager", self.waymo_traffic_manager)
        else:
            self.is_current_real_data = False
            self.engine.register_manager("traffic_manager", self.pg_traffic_manager)
            self.engine.register_manager("map_manager", self.pg_map_manager)
            self._init_pg_episode()
        self.engine.accept("s", self.stop)
        self.engine.accept("q", self.switch_to_third_person_view)
        self.engine.accept("b", self.switch_to_top_down_view)

    def change_suite(self):
        if engine_initialized():
            # must lazy initialize at first
            if get_np_random(None).rand() < self.real_data_ratio:
                # change to real environment
                self.engine.update_manager("map_manager", self.waymo_map_manager, destroy_previous_manager=False)
                self.engine.update_manager(
                    "traffic_manager", self.waymo_traffic_manager, destroy_previous_manager=False
                )
                self.is_current_real_data = True
            else:
                self.is_current_real_data = False
                # change to PG environment
                self.engine.update_manager("map_manager", self.pg_map_manager, destroy_previous_manager=False)
                self.engine.update_manager("traffic_manager", self.pg_traffic_manager, destroy_previous_manager=False)
                self._init_pg_episode()

    def _init_pg_episode(self):
        self.config["target_vehicle_configs"]["default_agent"]["spawn_lane_index"] = (
            FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, self.engine.np_random.randint(3)
        )
        self.config["target_vehicle_configs"]["default_agent"]["destination"] = None

    def reset(self, force_seed: Union[None, int] = None):
        self.change_suite()
        # ===== same as BaseEnv =====
        self.lazy_init()  # it only works the first time when reset() is called to avoid the error when render
        self._reset_global_seed(force_seed)
        if self.engine is None:
            raise ValueError(
                "Current MetaDrive instance is broken. Please make sure there is only one active MetaDrive "
                "environment exists in one process. You can try to call env.close() and then call "
                "env.reset() to rescue this environment. However, a better and safer solution is to check the "
                "singleton of MetaDrive and restart your program."
            )
        self.engine.reset()
        if self._top_down_renderer is not None:
            self._top_down_renderer.reset(self.current_map)

        self.dones = {agent_id: False for agent_id in self.vehicles.keys()}
        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)

        assert (len(self.vehicles) == self.num_agents) or (self.num_agents == -1)
        # ^^^^^^ same as Base Env ^^^^^

        if not self.is_current_real_data:
            # give a initial speed when on metadrive
            self.vehicle.set_velocity(self.vehicle.heading, self.engine.np_random.randint(10))
        return self._get_reset_return()

    def _reset_global_seed(self, force_seed=None):
        current_seed = force_seed if force_seed is not None else get_np_random(None).randint(
            self.config["start_case_index"], self.config["start_case_index"] +
            self.config["case_num"] if self.is_current_real_data else self.config["environment_num"]
        )
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
            return vehicle.on_yellow_continuous_line or vehicle.crash_sidewalk or (not vehicle.on_lane)


class MixWaymoPGEnvWrapper(MixWaymoPGEnv):
    """
    This class is a convinient interface receive
    {"real_data_ratio": xyz,
    "total_case_num":xyz} as input
    """
    TOTAL_CASE = 100  # default 100

    def __init__(self, config=None):
        assert "waymo_data_directory" in config, "tell me waymo data path please"
        assert "real_data_ratio" in config, "tell me waymo data path please"
        env_config = config.copy()
        ratio = config["real_data_ratio"]
        assert 0 <= ratio <= 1, "ratio should be in [0, 1]"
        env_config["case_num"] = int(config.get("total_case_num", self.TOTAL_CASE) * ratio)
        env_config["environment_num"] = int(config.get("total_case_num", self.TOTAL_CASE) - env_config["case_num"])
        if "real_data_ratio" in env_config:
            env_config.pop("real_data_ratio")
        if "total_case_num" in env_config:
            env_config.pop("total_case_num")
        super(MixWaymoPGEnvWrapper, self).__init__(env_config)


if __name__ == "__main__":
    env = MixWaymoPGEnvWrapper(
        dict(
            manual_control=True,
            use_render=True,
            waymo_data_directory="E:\\PAMI_waymo_data\\idm_filtered\\validation",
            # case_num=2,
            # # start_case=32,
            # environment_num=0
            # total_case_num=10,
            real_data_ratio=0.3
        )
    )
    env.reset()
    while True:
        o, r, d, i = env.step(env.action_space.sample())
        env.render(text={"ts": env.episode_step})
        if d:
            env.reset()
