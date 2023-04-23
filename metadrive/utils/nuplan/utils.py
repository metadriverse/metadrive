import os
import tempfile
from dataclasses import dataclass
from os.path import join
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from metadrive.utils import is_win

try:
    import geopandas as gpd
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
    from shapely.ops import unary_union
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
    import hydra
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
    from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
    from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
    from nuplan.planning.script.utils import set_up_common_builder
    import nuplan
except ImportError:
    logger.warning("Can not import nuplan-devkit")

NUPLAN_PACKAGE_PATH = os.path.dirname(nuplan.__file__)


def convert_one_scenario(scenario: NuPlanScenario):



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


@dataclass
class HydraConfigPaths:
    """
    Stores relative hydra paths to declutter tutorial.
    """

    common_dir: str
    config_name: str
    config_path: str
    experiment_dir: str
