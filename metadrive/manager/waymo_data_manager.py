import os

from metadrive.manager.base_manager import BaseManager
from metadrive.utils.waymo_utils.waymo_utils import read_waymo_data
from tqdm import tqdm
from collections import deque


class WaymoDataManager(BaseManager):
    def __init__(self, save_memory=False, max_len=100):
        super(WaymoDataManager, self).__init__()
        self.directory = self.engine.global_config["waymo_data_directory"]
        self.case_num = self.engine.global_config["case_num"]
        self.start_case_index = self.engine.global_config["start_case_index"]
        self.waymo_case = {}

        self.save_memory = save_memory
        if self.save_memory:
            self.loaded_case = deque(maxlen=max_len)
            self.max_len = max_len

        for i in tqdm(range(self.start_case_index, self.start_case_index + self.case_num), desc="Check Waymo Data"):
            p = os.path.join(self.directory, "{}.pkl".format(i))
            assert os.path.exists(p), "No Data {} at path: {}".format(i, p)

            if self.save_memory:
                pass
            else:
                self.waymo_case[i] = self._get_case(i)

    def _get_case(self, i):
        assert self.start_case_index <= i < self.start_case_index + self.case_num, \
            "Case ID exceeds range"
        file_path = os.path.join(self.directory, "{}.pkl".format(i))
        return read_waymo_data(file_path)

    def get_case(self, i):
        if self.save_memory:
            if i in self.loaded_case:
                return self.waymo_case[i]

            # i-th case is not loaded yet
            while len(self.loaded_case) >= self.max_len:
                should_remove = self.loaded_case.popleft()
                self.waymo_case.pop(should_remove)

            self.waymo_case[i] = self._get_case(i)
            return self.waymo_case[i]

        else:
            return self.waymo_case[i]
