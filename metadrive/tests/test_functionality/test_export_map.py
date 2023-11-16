from metadrive.engine.asset_loader import AssetLoader
from metadrive.scenario.utils import draw_map
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.idm_policy import TrajectoryIDMPolicy


def test_export_waymo_map(render=False):
    TrajectoryIDMPolicy.NORMAL_SPEED = 30
    asset_path = AssetLoader.asset_path
    env = ScenarioEnv(
        {
            "manual_control": False,
            "no_traffic": True,
            "use_render": False,
            "data_directory": AssetLoader.file_path(asset_path, "waymo", unix_style=False),
            "num_scenarios": 3
        }
    )
    try:
        for seed in range(3):
            env.reset(seed=seed)
            map_vector = env.current_map.get_map_features()
            draw_map(map_vector, True if render else False)
    finally:
        env.close()


def test_metadrive_map_export(render=False):
    env = MetaDriveEnv(dict(image_observation=False, map=6, num_scenarios=1, start_seed=0))
    try:
        env.reset(seed=0)
        map_vector = env.current_map.get_map_features()
        draw_map(map_vector, True if render else False)
    finally:
        env.close()


if __name__ == "__main__":
    test_export_waymo_map(True)
