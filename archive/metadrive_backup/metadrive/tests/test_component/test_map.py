import cv2

from metadrive.component.map.scenario_map import ScenarioMap
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.engine_utils import initialize_engine, close_engine
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.manager.scenario_data_manager import ScenarioDataManager


# @pytest.mark.parametrize("dir", ["waymo"])
def test_map_get_semantic_map(dir="waymo", render=False, show=False):
    default_config = ScenarioEnv.default_config()
    default_config["use_render"] = render
    default_config["debug"] = False
    default_config["debug_static_world"] = False
    default_config["data_directory"] = AssetLoader.file_path(dir, unix_style=False)
    # default_config["data_directory"] = AssetLoader.file_path("nuscenes", unix_style=False)
    # default_config["data_directory"] = "/home/shady/Downloads/test_processed"
    default_config["num_scenarios"] = 3
    engine = initialize_engine(default_config)
    try:
        size, res = 512, 2
        engine.data_manager = ScenarioDataManager()
        for idx in range(default_config["num_scenarios"]):
            engine.seed(idx)
            m_data = engine.data_manager.get_scenario(idx, should_copy=False)["map_features"]
            map = ScenarioMap(map_index=idx, map_data=m_data)
            heightfield = map.get_semantic_map([0, 0], size, res)
            assert heightfield.shape[0] == heightfield.shape[1] == int(size * res)
            if show:
                cv2.imshow('terrain', heightfield)
                cv2.waitKey(0)
    finally:
        close_engine()


def test_map_get_elevation_map(dir="waymo", render=False, show=False):
    default_config = ScenarioEnv.default_config()
    default_config["use_render"] = render
    default_config["debug"] = False
    default_config["debug_static_world"] = False
    default_config["data_directory"] = AssetLoader.file_path(dir, unix_style=False)
    # default_config["data_directory"] = AssetLoader.file_path("nuscenes", unix_style=False)
    # default_config["data_directory"] = "/home/shady/Downloads/test_processed"
    default_config["num_scenarios"] = 3
    engine = initialize_engine(default_config)
    try:
        size, res = 1024, 1
        engine.data_manager = ScenarioDataManager()
        for idx in range(default_config["num_scenarios"]):
            engine.seed(idx)
            m_data = engine.data_manager.get_scenario(idx, should_copy=False)["map_features"]
            map = ScenarioMap(map_index=idx, map_data=m_data)
            heightfield = map.get_height_map([0, 0], size, res, extension=4)
            assert heightfield.shape[0] == heightfield.shape[1] == int(size * res)
            if show:
                cv2.imshow('terrain', heightfield)
                cv2.waitKey(0)
    finally:
        close_engine()


if __name__ == "__main__":
    # test_map_get_semantic_map("waymo", render=False, show=True)
    test_map_get_elevation_map("waymo", render=False, show=True)
