import logging
import os
import tempfile
from os.path import join
import hydra
import nuplan.planning.script.builders.worker_pool_builder
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.script.utils import set_up_common_builder
from dataclasses import dataclass
from metadrive.manager.base_manager import BaseManager
from metadrive.utils.utils import is_win

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(__file__)


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


class NuPlanDataManager(BaseManager):
    """
    This manager serves as the interface between nuplan-devkit and MetaDrive
    """
    NUPLAN_PACKAGE_PATH = os.path.dirname(nuplan.__file__)
    NUPLAN_ROOT_PATH = os.path.dirname(NUPLAN_PACKAGE_PATH)
    NUPLAN_TUTORIAL_PATH = os.path.join(os.path.dirname(nuplan.__file__), "tutorial")

    def __init__(self):
        super(NuPlanDataManager, self).__init__()
        logger.info("\n \n ############### Start Loading NuPlan Data ############### \n")
        # get original nuplan cfg
        cfg = self._get_nuplan_cfg()
        profiler_name = 'building_simulation'
        common_builder = set_up_common_builder(cfg=cfg, profiler_name=profiler_name)

        # Build scenario builder
        scenario_builder = build_scenario_builder(cfg=cfg)
        scenario_filter = build_scenario_filter(cfg.scenario_filter)

        # get scenarios from database
        self._nuplan_scenarios = scenario_builder.get_scenarios(scenario_filter, common_builder.worker)

        # filter scenario according to config
        self.start_scenario_index = self.engine.global_config["start_scenario_index"]
        self._num_scenarios = self.engine.global_config["num_scenarios"]
        assert len(self._nuplan_scenarios) >= self.start_scenario_index + self.num_scenarios, \
            "Number of scenes are not enough, " \
            "\n num nuplan scenarios: {}" \
            "\n start_scenario_index: {}" \
            "\n scenario num: {}".format(len(self._nuplan_scenarios), self.start_scenario_index, self.num_scenarios)
        logger.info("\n \n ############### Finish Loading NuPlan Data ############### \n")

        self._current_scenario_index = None

    @property
    def time_interval(self):
        return self.current_scenario.database_interval

    @property
    def num_scenarios(self):
        return self._num_scenarios

    @property
    def current_scenario_index(self):
        return self._current_scenario_index

    @property
    def current_scenario(self):
        return self._nuplan_scenarios[self._current_scenario_index]

    def _get_nuplan_cfg(self):
        BASE_CONFIG_PATH = os.path.join(self.NUPLAN_PACKAGE_PATH, "planning", "script")
        simulation_hydra_paths = construct_simulation_hydra_paths(BASE_CONFIG_PATH)

        # Initialize configuration management system
        hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
        hydra.initialize_config_dir(config_dir=simulation_hydra_paths.config_path)

        SAVE_DIR = tempfile.mkdtemp()
        EGO_CONTROLLER = 'perfect_tracking_controller'  # [log_play_back_controller, perfect_tracking_controller]
        OBSERVATION = 'box_observation'  # [box_observation, idm_agents_observation, lidar_pc_observation]

        DATASET_PARAMS = self.engine.global_config["DATASET_PARAMS"]

        # Compose the configuration
        overrides = [
            f'group={SAVE_DIR}',
            'worker=sequential',
            f'ego_controller={EGO_CONTROLLER}',
            f'observation={OBSERVATION}',
            f'hydra.searchpath=[{simulation_hydra_paths.common_dir}, {simulation_hydra_paths.experiment_dir}]',
            'output_dir=${group}/${experiment}',
            *DATASET_PARAMS,
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
        cfg = hydra.compose(config_name=simulation_hydra_paths.config_name, overrides=overrides)
        return cfg

    def get_scenario(self, index, force_get_current_scenario=True):
        assert self.start_scenario_index <= index < self.start_scenario_index + self.num_scenarios
        if force_get_current_scenario:
            assert index == self.random_seed
            return self.current_scenario
        else:
            return self._nuplan_scenarios[index]

    def seed(self, random_seed):
        assert self.start_scenario_index <= random_seed < self.start_scenario_index + self.num_scenarios
        super(NuPlanDataManager, self).seed(random_seed)
        self._current_scenario_index = random_seed

    @property
    def current_scenario_length(self):
        return self.current_scenario.get_number_of_iterations()

    def get_metadata(self):
        state = super(NuPlanDataManager, self).get_metadata()
        raw_data = self.current_scenario
        state["raw_data"] = raw_data
        return state
