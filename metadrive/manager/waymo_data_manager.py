import os

from metadrive.manager.base_manager import BaseManager
from metadrive.utils.waymo_utils.waymo_utils import read_waymo_data
from tqdm import tqdm


class WaymoDataManager(BaseManager):
    def __init__(self):
        super(WaymoDataManager, self).__init__()
        self.directory = self.engine.global_config["waymo_data_directory"]
        self.case_num = self.engine.global_config["case_num"]
        self.start_case_index = self.engine.global_config["start_case_index"]
        for i in tqdm(range(self.case_num), desc="Check Data"):
            assert os.path.exists(os.path.join(self.directory, "{}.pkl".format(i + self.start_case_index))
                                  ), "No Data {}".format(i + self.start_case_index)

    def get_case(self, i):
        assert self.start_case_index <= i + self.start_case_index < self.start_case_index + self.case_num, \
            "Case ID exceeds range"
        file_path = os.path.join(self.directory, "{}.pkl".format(i + self.start_case_index))
        return read_waymo_data(file_path)
