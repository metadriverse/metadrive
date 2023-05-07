"""
This script aims to convert nuplan scenarios to ScenarioDescription, so that we can load any nuplan scenarios into
MetaDrive.
"""
import copy
import os
import pickle
import shutil

import tqdm

from metadrive.engine.asset_loader import AssetLoader
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.utils.nuplan.utils import get_nuplan_scenarios, convert_one_scenario
from metadrive.utils.utils import dict_recursive_remove_array

import logging

logger = logging.getLogger(__name__)

logger.warning(
    "This converters will de deprecated. "
    "Use tools in ScenarioNet instead: https://github.com/metadriverse/ScenarioNet."
)


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
    summary_file = ScenarioDescription.DATASET.SUMMARY_FILE
    if worker_index is not None:
        desc += "Worker {} ".format(worker_index)
        summary_file = "dataset_summary_worker{}.pkl".format(worker_index)

    # Init.
    scenarios = get_nuplan_scenarios(dataset_params)
    for scenario in tqdm.tqdm(scenarios):
        sd_scenario = convert_one_scenario(scenario)
        sd_scenario = sd_scenario.to_dict()
        ScenarioDescription.sanity_check(sd_scenario, check_self_type=True)
        export_file_name = ScenarioDescription.get_export_file_name("nuplan", "v1.1", scenario.scenario_name)
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
    if delay_remove is not None:
        assert delay_remove == save_path, delay_remove + " vs. " + save_path


if __name__ == "__main__":
    # 14 types
    all_scenario_types = "[behind_pedestrian_on_pickup_dropoff,  \
                            near_multiple_vehicles, \
                            high_magnitude_jerk, \
                            crossed_by_vehicle, \
                            following_lane_with_lead, \
                            changing_lane_to_left, \
                            accelerating_at_traffic_light_without_lead, \
                            stopping_at_stop_sign_with_lead, \
                            traversing_narrow_lane, \
                            waiting_for_pedestrian_to_cross, \
                            starting_left_turn, \
                            starting_high_speed_turn, \
                            starting_unprotected_cross_turn, \
                            starting_protected_noncross_turn, \
                            on_pickup_dropoff]"

    dataset_params = [
        # builder setting
        "scenario_builder=nuplan_mini",
        "scenario_builder.scenario_mapping.subsample_ratio_override=0.5",  # 10 hz

        # filter
        "scenario_filter=all_scenarios",  # simulate only one log
        "scenario_filter.remove_invalid_goals=true",
        "scenario_filter.shuffle=true",
        "scenario_filter.log_names=['2021.07.16.20.45.29_veh-35_01095_01486']",
        # "scenario_filter.scenario_types={}".format(all_scenario_types),
        # "scenario_filter.scenario_tokens=[]",
        # "scenario_filter.map_names=[]",
        # "scenario_filter.num_scenarios_per_type=1",
        # "scenario_filter.limit_total_scenarios=1000",
        # "scenario_filter.expand_scenarios=true",
        # "scenario_filter.limit_scenarios_per_type=10",  # use 10 scenarios per scenario type
        "scenario_filter.timestamp_threshold_s=20",  # minial scenario duration (s)
    ]
    output_path = AssetLoader.file_path("nuplan", return_raw_style=False)
    worker_index = None
    force_overwrite = True
    convert_scenarios(output_path, dataset_params, worker_index=worker_index, force_overwrite=force_overwrite)
