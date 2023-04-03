import copy
import os

from tqdm import tqdm

from metadrive.manager.base_manager import BaseManager
from metadrive.scenario.scenario_description import ScenarioDescription as SD, MetaDriveType
from metadrive.scenario.utils import read_scenario_data
from metadrive.utils.data_buffer import DataBuffer


class ScenarioDataManager(BaseManager):
    DEFAULT_DATA_BUFFER_SIZE = 100

    def __init__(self):
        super(ScenarioDataManager, self).__init__()

        from metadrive.engine.engine_utils import get_engine
        engine = get_engine()

        store_map = engine.global_config.get("store_map", False)
        store_map_buffer_size = engine.global_config.get("store_map_buffer_size", self.DEFAULT_DATA_BUFFER_SIZE)
        self.directory = engine.global_config["data_directory"]
        self.num_scenarios = engine.global_config["num_scenarios"]
        self.start_scenario_index = engine.global_config["start_scenario_index"]

        self.store_map = store_map
        self._scenario = DataBuffer(store_map_buffer_size if self.store_map else self.num_scenarios)

        for i in tqdm(range(self.start_scenario_index, self.start_scenario_index + self.num_scenarios),
                      desc="Check Scenario Data"):
            p = os.path.join(self.directory, "{}.pkl".format(i))
            assert os.path.exists(p), "No Data {} at path: {}".format(i, p)

            # if self.store_map:
            # If we wish to store map (which requires huge memory), we load data immediately to exchange effiency
            # later
            # self._scenario[i] = self._get_scenario(i)

    def _get_scenario(self, i):
        assert self.start_scenario_index <= i < self.start_scenario_index + self.num_scenarios, \
            "scenario ID exceeds range"
        file_path = os.path.join(self.directory, "{}.pkl".format(i))
        ret = read_scenario_data(file_path)
        assert isinstance(ret, SD)
        return ret

    def get_scenario(self, i, should_copy=False):

        _debug_memory_leak = False

        if i not in self._scenario:

            if _debug_memory_leak:
                # inner psutil function
                def process_memory():
                    import psutil
                    import os
                    process = psutil.Process(os.getpid())
                    mem_info = process.memory_info()
                    return mem_info.rss

                cm = process_memory()

            self._scenario.clear_if_necessary()

            if _debug_memory_leak:
                lm = process_memory()
                print("{}:  Reset! Mem Change {:.3f}MB".format("data manager clear scenario", (lm - cm) / 1e6))
                cm = lm

            # print("===Getting new scenario: ", i)
            self._scenario[i] = self._get_scenario(i)

            if _debug_memory_leak:
                lm = process_memory()
                print("{}:  Reset! Mem Change {:.3f}MB".format("data manager read scenario", (lm - cm) / 1e6))
                cm = lm

        else:
            pass
            # print("===Don't need to get new scenario. Just return: ", i)

        if should_copy:
            return copy.deepcopy(self._scenario[i])

        # Data Manager is the first manager that accesses  data.
        # It is proper to let it validate the metadata and change the global config if needed.
        ret = self._scenario[i]
        self.validate_data(ret)

        return ret

    def get_state(self):
        raw_data = self.get_scenario(self.engine.global_seed)
        state = super(ScenarioDataManager, self).get_state()
        state["raw_data"] = raw_data
        return state

    def validate_data(self, scenario):
        # It supports loading WaymoData or exported data in two coordinates
        if scenario[SD.METADATA][SD.COORDINATE] == MetaDriveType.COORDINATE_WAYMO:
            self.engine.global_config["coordinate_transform"] = True
        elif scenario[SD.METADATA][SD.COORDINATE] == MetaDriveType.COORDINATE_METADRIVE:
            self.engine.global_config["coordinate_transform"] = False
        else:
            raise ValueError()
