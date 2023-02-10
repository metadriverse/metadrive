import copy
import os

from tqdm import tqdm

from metadrive.manager.base_manager import BaseManager
from metadrive.utils.data_buffer import DataBuffer
from metadrive.utils.waymo_utils.waymo_utils import read_waymo_data


class WaymoDataManager(BaseManager):
    DEFAULT_DATA_BUFFER_SIZE = 100

    def __init__(self):
        super(WaymoDataManager, self).__init__()
        store_map = self.engine.global_config.get("store_map", False)
        store_map_buffer_size = self.engine.global_config.get("store_map_buffer_size", self.DEFAULT_DATA_BUFFER_SIZE)
        self.directory = self.engine.global_config["waymo_data_directory"]
        self.case_num = self.engine.global_config["case_num"]
        self.start_case_index = self.engine.global_config["start_case_index"]

        self.store_map = store_map
        self.waymo_case = DataBuffer(store_map_buffer_size if self.store_map else self.case_num)

        for i in tqdm(range(self.start_case_index, self.start_case_index + self.case_num), desc="Check Waymo Data"):
            p = os.path.join(self.directory, "{}.pkl".format(i))
            assert os.path.exists(p), "No Data {} at path: {}".format(i, p)

            # if self.store_map:
            # If we wish to store map (which requires huge memory), we load data immediately to exchange effiency
            # later
            # self.waymo_case[i] = self._get_case(i)

    def _get_case(self, i):
        assert self.start_case_index <= i < self.start_case_index + self.case_num, \
            "Case ID exceeds range"
        file_path = os.path.join(self.directory, "{}.pkl".format(i))
        return copy.deepcopy(read_waymo_data(file_path))

    def get_case(self, i):
        if i not in self.waymo_case:
            # inner psutil function
            # def process_memory():
            #     import psutil
            #     import os
            #     process = psutil.Process(os.getpid())
            #     mem_info = process.memory_info()
            #     return mem_info.rss
            #
            # cm = process_memory()

            self.waymo_case.clear_if_necessary()

            # lm = process_memory()
            # print("{}:  Reset! Mem Change {:.3f}MB".format("data manager 1", (lm - cm) / 1e6))
            # cm = lm

            self.waymo_case[i] = self._get_case(i)

            # lm = process_memory()
            # print("{}:  Reset! Mem Change {:.3f}MB".format("data manager 2", (lm - cm) / 1e6))
            # cm = lm

        # return copy.deepcopy(self.waymo_case[i])
        return self.waymo_case[i]
