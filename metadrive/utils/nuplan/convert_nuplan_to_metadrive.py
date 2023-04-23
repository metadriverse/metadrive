"""
This script aims to convert nuplan scenarios to ScenarioDescription, so that we can load any nuplan scenarios into
MetaDrive.
"""
import copy
import logging
import os
import pickle
import shutil

import tqdm

from metadrive.engine.asset_loader import AssetLoader
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.utils.nuplan.utils import get_nuplan_scenarios, convert_one_scenario
from metadrive.utils.utils import dict_recursive_remove_array


def convert_scenarios(output_path, dataset_params, worker_index=None, force_overwrite=False):
    save_path = copy.deepcopy(output_path)
    output_path = output_path + "_tmp"
    # meta recorder and data summary
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=False)

    # make real save dir
    delay_remove = None
    if os.path.exists(save_path):
        if force_overwrite:
            delay_remove = save_path
        else:
            raise ValueError("Directory already exists! Abort")

    metadata_recorder = {}
    total_scenarios = 0
    desc = ""
    summary_file = "dataset_summary.pkl"
    if worker_index is not None:
        desc += "Worker {} ".format(worker_index)
        summary_file = "dataset_summary_worker{}.pkl".format(worker_index)

    # Init.
    scenarios = get_nuplan_scenarios(dataset_params)
    for scenario in tqdm.tqdm(scenarios):
        sd_scenario = convert_one_scenario(scenario)
        sd_scenario = sd_scenario.to_dict()
        ScenarioDescription.sanity_check(sd_scenario, check_self_type=True)
        # TODO Naming
        export_file_name = "sd_{}_{}.pkl".format("nuplan_", "test")
        p = os.path.join(output_path, export_file_name)
        with open(p, "wb") as f:
            pickle.dump(sd_scenario, f)
        metadata_recorder[export_file_name] = copy.deepcopy(sd_scenario[ScenarioDescription.METADATA])
    # rename and save
    if delay_remove is not None:
        shutil.rmtree(delay_remove)
    os.rename(output_path, save_path)
    summary_file = os.path.join(save_path, summary_file)
    with open(summary_file, "wb") as file:
        pickle.dump(dict_recursive_remove_array(metadata_recorder), file)
    print("Summary is saved at: {}".format(summary_file))
    assert delay_remove == save_path


if __name__ == "__main__":
    dataset_params = [
        'scenario_builder=nuplan_mini',
        # use nuplan mini database (2.5h of 8 autolabeled logs in Las Vegas)
        'scenario_filter=one_continuous_log',  # simulate only one log
        "scenario_filter.log_names=['2021.07.16.20.45.29_veh-35_01095_01486']",
        'scenario_filter.limit_total_scenarios=2800',  # use 2 total scenarios
    ]
    output_path = AssetLoader.file_path("nuscenes", return_raw_style=False)
    worker_index = None
    force_overwrite = True
    convert_scenarios(output_path, dataset_params, worker_index=worker_index, force_overwrite=force_overwrite)
