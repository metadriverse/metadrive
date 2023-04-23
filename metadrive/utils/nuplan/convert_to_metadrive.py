"""
This script aims to convert nuplan scenarios to ScenarioDescription, so that we can load any nuplan scenarios into
MetaDrive.
"""
try:
    import geopandas as gpd
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
    from shapely.ops import unary_union
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
except ImportError:
    pass
import copy
import logging
import os
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from os.path import join

import hydra
import nuplan.planning.script.builders.worker_pool_builder
import tqdm
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.script.utils import set_up_common_builder

from metadrive.engine.asset_loader import AssetLoader
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.utils.utils import dict_recursive_remove_array
from metadrive.utils.utils import is_win

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUPLAN_PACKAGE_PATH = os.path.dirname(nuplan.__file__)


# Copied from nuplan-devkit tutorial_utils.py
@dataclass
class HydraConfigPaths:
    """
    Stores relative hydra paths to declutter tutorial.
    """

    common_dir: str
    config_name: str
    config_path: str
    experiment_dir: str


def construct_simulation_hydra_paths(base_config_path: str):
    """
    Specifies relative paths to simulation configs to pass to hydra to declutter tutorial.
    :param base_config_path: Base config path.
    :return: Hydra config path.
    """
    common_dir = "file://" + join(base_config_path, 'config', 'common')
    config_name = 'default_simulation'
    config_path = join(base_config_path, 'config', 'simulation')
    experiment_dir = "file://" + join(base_config_path, 'experiments')
    return HydraConfigPaths(common_dir, config_name, config_path, experiment_dir)


def get_nuplan_scenarios(dataset_parameters, nuplan_package_path=NUPLAN_PACKAGE_PATH):
    """
    Return a list of nuplan scenarios according to dataset_parameters
    """
    base_config_path = os.path.join(nuplan_package_path, "planning", "script")
    simulation_hydra_paths = construct_simulation_hydra_paths(base_config_path)

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize_config_dir(config_dir=simulation_hydra_paths.config_path)

    save_dir = tempfile.mkdtemp()
    ego_controller = 'perfect_tracking_controller'  # [log_play_back_controller, perfect_tracking_controller]
    observation = 'box_observation'  # [box_observation, idm_agents_observation, lidar_pc_observation]

    # Compose the configuration
    overrides = [
        f'group={save_dir}',
        'worker=sequential',
        f'ego_controller={ego_controller}',
        f'observation={observation}',
        f'hydra.searchpath=[{simulation_hydra_paths.common_dir}, {simulation_hydra_paths.experiment_dir}]',
        'output_dir=${group}/${experiment}',
        *dataset_parameters,
    ]
    if is_win():
        overrides.extend(
            [
                f'job_name=planner_tutorial',
                'experiment=${experiment_name}/${job_name}/${experiment_time}',
            ]
        )
    else:
        overrides.append(f'experiment_name=planner_tutorial')

    # get config
    cfg = hydra.compose(config_name=simulation_hydra_paths.config_name, overrides=overrides)

    profiler_name = 'building_simulation'
    common_builder = set_up_common_builder(cfg=cfg, profiler_name=profiler_name)

    # Build scenario builder
    scenario_builder = build_scenario_builder(cfg=cfg)
    scenario_filter = build_scenario_filter(cfg.scenario_filter)

    # get scenarios from database
    return scenario_builder.get_scenarios(scenario_filter, common_builder.worker)


def convert_one_scenario(scenario: NuPlanScenario):
    return {}


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
