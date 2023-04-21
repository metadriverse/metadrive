from metadrive.engine.asset_loader import AssetLoader

try:
    from metadrive.envs.real_data_envs.nuplan_env import NuPlanEnv
except ImportError:
    pass
from metadrive.scenario.utils import draw_map
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.policy.idm_policy import WaymoIDMPolicy


def test_export_waymo_map(render=False):
    WaymoIDMPolicy.NORMAL_SPEED = 30
    asset_path = AssetLoader.asset_path
    env = WaymoEnv(
        {
            "manual_control": False,
            "no_traffic": True,
            "use_render": False,
            "data_directory": AssetLoader.file_path(asset_path, "waymo", return_raw_style=False),
            "num_scenarios": 3
        }
    )
    try:
        for seed in range(3):
            env.reset(force_seed=seed)
            map_vector = env.current_map.get_map_features()
            draw_map(map_vector, True if render else False)
    finally:
        env.close()


def test_metadrive_map_export(render=False):
    env = MetaDriveEnv(dict(image_observation=False, map=6, num_scenarios=1, start_seed=0))
    try:
        env.reset(force_seed=0)
        map_vector = env.current_map.get_map_features()
        draw_map(map_vector, True if render else False)
    finally:
        env.close()


def _test_nuplan_map_export(render=False):
    env = NuPlanEnv(
        {
            "DATASET_PARAMS": [
                'scenario_builder=nuplan_mini',
                'scenario_filter=one_continuous_log',  # simulate only one log
                "scenario_filter.log_names=['2021.09.16.15.12.03_veh-42_01037_01434']",
                'scenario_filter.limit_total_scenarios=1000',  # use 2 total scenarios
            ]
        }
    )
    try:
        env.reset(force_seed=0)
        map_vector = env.current_map.get_map_features()
        draw_map(map_vector, True if render else False)
    finally:
        env.close()


if __name__ == "__main__":
    # test_export_waymo_map(True)
    _test_nuplan_map_export(True)
