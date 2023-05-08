import os.path
from distutils.dir_util import copy_tree

from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.scenario.utils import read_dataset_summary, read_scenario_data


def test_read_waymo_data():
    summary_dict, summary_list, mapping = read_dataset_summary(AssetLoader.file_path("waymo", return_raw_style=False))
    for p in summary_list:
        data = read_scenario_data(AssetLoader.file_path("waymo", mapping[p], p, return_raw_style=False))
        data.sanity_check(data, check_self_type=False, valid_check=False)
        print("Finish: ", p)


def test_read_data_no_summary():
    dir = AssetLoader.file_path("waymo", return_raw_style=False)
    new_dir = "test_read_copy_waymo"

    #  make fake dataset
    copy_tree(dir, new_dir)
    os.remove(os.path.join("test_read_copy_waymo", ScenarioDescription.DATASET.SUMMARY_FILE))
    env = ScenarioEnv(
        {
            "manual_control": True,
            "no_traffic": False,
            "start_scenario_index": 0,
            "show_coordinates": True,
            "num_scenarios": 3,
            "data_directory": new_dir,
        }
    )
    try:
        env.reset()
    finally:
        env.close()
