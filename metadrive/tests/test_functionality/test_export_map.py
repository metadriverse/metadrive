from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.policy.idm_policy import WaymoIDMPolicy


def test_export_waymo_map():
    WaymoIDMPolicy.NORMAL_SPEED = 30
    asset_path = AssetLoader.asset_path
    try:
        env = WaymoEnv(
            {
                "manual_control": False,
                "replay": True,
                "no_traffic": True,
                "use_render": False,
                "waymo_data_directory": AssetLoader.file_path(asset_path, "waymo", return_raw_style=False),
                "case_num": 3
            }
        )
        for seed in range(3):
            env.reset(force_seed=seed)
            env.current_map.get_map_vector()
    finally:
        env.close()
