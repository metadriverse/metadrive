import copy
import os
from metadrive.utils.config import Config
import numpy as np

from metadrive.manager.base_manager import BaseManager
from metadrive.scenario.scenario_description import ScenarioDescription as SD, MetaDriveType
from metadrive.scenario.utils import read_scenario_data, read_dataset_summary


class ScenarioDataManager(BaseManager):
    DEFAULT_DATA_BUFFER_SIZE = 100
    PRIORITY = -10

    def __init__(self):
        super(ScenarioDataManager, self).__init__()
        from metadrive.engine.engine_utils import get_engine
        engine = get_engine()

        self.store_data = engine.global_config["store_data"]
        self.directory = engine.global_config["data_directory"]
        self.num_scenarios = engine.global_config["num_scenarios"]
        self.start_scenario_index = engine.global_config["start_scenario_index"]

        # for multi-worker
        self.worker_index = self.engine.global_config["worker_index"]
        self.available_scenario_indices = [
            i for i in range(
                self.start_scenario_index + self.worker_index, self.start_scenario_index +
                self.num_scenarios, self.engine.global_config["num_workers"]
            )
        ]
        self._scenarios = {}

        # Read summary file first:
        self.summary_dict, self.summary_lookup, self.mapping = read_dataset_summary(self.directory)
        self.summary_lookup[:self.start_scenario_index] = [None] * self.start_scenario_index
        end_idx = self.start_scenario_index + self.num_scenarios
        self.summary_lookup[end_idx:] = [None] * (len(self.summary_lookup) - end_idx)

        # sort scenario for curriculum training
        self.scenario_difficulty = None
        self.sort_scenarios()

        # existence check
        assert self.start_scenario_index < len(self.summary_lookup), "Insufficient scenarios!"
        assert self.start_scenario_index + self.num_scenarios <= len(self.summary_lookup), \
            "Insufficient scenarios! Need: {} Has: {}".format(self.num_scenarios,
                                                              len(self.summary_lookup) - self.start_scenario_index)

        for p in self.summary_lookup[self.start_scenario_index:end_idx]:
            p = os.path.join(self.directory, self.mapping[p], p)
            assert os.path.exists(p), "No Data at path: {}".format(p)

        # stat
        self.coverage = [0 for _ in range(self.num_scenarios)]

    @property
    def current_scenario_summary(self):
        return self.current_scenario[SD.METADATA]

    def _get_scenario(self, i):
        assert i in self.available_scenario_indices, \
            "scenario index exceeds range, scenario index: {}, worker_index: {}".format(i, self.worker_index)
        assert i < len(self.summary_lookup)
        scenario_id = self.summary_lookup[i]
        file_path = os.path.join(self.directory, self.mapping[scenario_id], scenario_id)
        ret = read_scenario_data(file_path)
        assert isinstance(ret, SD)
        self.coverage[i - self.start_scenario_index] = 1
        return ret

    def before_reset(self):
        if not self.store_data:
            assert len(self._scenarios) <= 1, "It seems you access multiple scenarios in one episode"
            self._scenarios = {}

    def get_scenario(self, i, should_copy=False):

        _debug_memory_leak = False

        if i not in self._scenarios:

            if _debug_memory_leak:
                # inner psutil function
                def process_memory():
                    import psutil
                    import os
                    process = psutil.Process(os.getpid())
                    mem_info = process.memory_info()
                    return mem_info.rss

                cm = process_memory()

            # self._scenarios.clear_if_necessary()

            if _debug_memory_leak:
                lm = process_memory()
                print("{}:  Reset! Mem Change {:.3f}MB".format("data manager clear scenario", (lm - cm) / 1e6))
                cm = lm

            ret = self._get_scenario(i)
            self._scenarios[i] = ret

            if _debug_memory_leak:
                lm = process_memory()
                print("{}:  Reset! Mem Change {:.3f}MB".format("data manager read scenario", (lm - cm) / 1e6))
                cm = lm

        else:
            ret = self._scenarios[i]
            # print("===Don't need to get new scenario. Just return: ", i)

        if should_copy:
            return copy.deepcopy(self._scenarios[i])

        # Data Manager is the first manager that accesses  data.
        # It is proper to let it validate the metadata and change the global config if needed.

        return ret

    def get_metadata(self):
        state = super(ScenarioDataManager, self).get_metadata()
        raw_data = self.current_scenario
        state["raw_data"] = raw_data
        return state

    @property
    def current_scenario_length(self):
        return self.current_scenario[SD.LENGTH]

    @property
    def current_scenario(self):
        return self.get_scenario(self.engine.global_random_seed)

    def sort_scenarios(self):
        """
        TODO(LQY): consider exposing this API to config
        Sort scenarios to support curriculum training. You are encouraged to customize your own sort method
        :return: sorted scenario list
        """
        if self.engine.max_level == 0:
            raise ValueError("Curriculum Level should be greater than 1")
        elif self.engine.max_level == 1:
            return

        def _score(scenario_id):
            file_path = os.path.join(self.directory, self.mapping[scenario_id], scenario_id)
            scenario = read_scenario_data(file_path)
            obj_weight = 0

            # calculate curvature
            ego_car_id = scenario[SD.METADATA][SD.SDC_ID]
            state_dict = scenario["tracks"][ego_car_id]["state"]
            valid_track = state_dict["position"][np.where(state_dict["valid"].astype(int))][..., :2]

            dir = valid_track[1:] - valid_track[:-1]
            dir = np.arctan2(dir[..., 1], dir[..., 0])
            curvature = sum(abs(dir[1:] - dir[:-1]) / np.pi) + 1

            sdc_moving_dist = SD.sdc_moving_dist(scenario)
            num_moving_objs = SD.num_moving_object(scenario, object_type=MetaDriveType.VEHICLE)
            return sdc_moving_dist * curvature + num_moving_objs * obj_weight

        start = self.start_scenario_index
        end = self.start_scenario_index + self.num_scenarios
        id_scores = [(s_id, _score(s_id)) for s_id in self.summary_lookup[start:end]]
        id_scores = sorted(id_scores, key=lambda scenario: scenario[-1])
        self.summary_lookup[start:end] = [id_score[0] for id_score in id_scores]
        self.scenario_difficulty = {id_score[0]: id_score[1] for id_score in id_scores}

    def clear_stored_scenarios(self):
        self._scenarios = {}

    @property
    def current_scenario_difficulty(self):
        return self.scenario_difficulty[self.summary_lookup[self.engine.global_random_seed]
                                        ] if self.scenario_difficulty is not None else 0

    @property
    def current_scenario_id(self):
        return self.current_scenario_summary["scenario_id"]

    @property
    def current_scenario_file_name(self):
        return self.summary_lookup[self.engine.global_random_seed]

    @property
    def data_coverage(self):
        return sum(self.coverage) / len(self.coverage) * self.engine.global_config["num_workers"]

    def destroy(self):
        """
        Clear memory
        """
        super(ScenarioDataManager, self).destroy()
        self._scenarios = {}
        Config.clear_nested_dict(self.summary_dict)
        self.summary_lookup.clear()
        self.mapping.clear()
        self.summary_dict, self.summary_lookup, self.mapping = None, None, None
